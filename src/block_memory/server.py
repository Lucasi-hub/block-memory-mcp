"""
Block Memory MCP Server — 区块化动态系数召回AI记忆架构

Implements: block storage + FTS5 coefficient recall + temporal decay + weight ceiling.
Each block has a type (chat/task/technical) with different decay rates.
Recall combines FTS5 text matching + temporal decay + recency + popularity.

Database path: $BLOCK_MEMORY_DB or ~/.block-memory/blocks.db (auto-created)
"""

import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ── Paths ────────────────────────────────────────────
DB_PATH = Path(os.environ.get("BLOCK_MEMORY_DB", Path.home() / ".block-memory" / "blocks.db"))
DB_DIR = DB_PATH.parent

# ── Static Base Config ───────────────────────────────
BASE_CONFIG_DEFAULTS = {
    "chat_decay_rate":      (0.05,  "Chat block decay rate (5%/h → ~24h to 30%)"),
    "task_decay_rate":       (0.02,  "Task block decay rate (2%/h → ~50h to 37%)"),
    "technical_decay_rate":  (0.005, "Technical block decay rate (0.5%/h → ~138h to 50%)"),
    "weight_ceiling":        (5.0,   "Hard upper limit for current_weight"),
    "recall_threshold":      (0.1,   "Composite score below which blocks are dropped"),
    "recall_boost":          (0.1,   "Weight increment on each successful recall"),
    "fts5_weight":           (0.6,   "FTS5 text match weight in composite score"),
    "recency_weight":        (0.2,   "Recency weight in composite score"),
    "popularity_weight":     (0.2,   "Popularity (recall count) weight in composite score"),
}

DECAY_RATES = {
    "chat":      "chat_decay_rate",
    "task":      "task_decay_rate",
    "technical": "technical_decay_rate",
}


# ── Database ──────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    """Open (or create) the database and ensure schema exists."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    db_exists = DB_PATH.exists()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    if not db_exists:
        _create_schema(conn)
    return conn


def _create_schema(conn: sqlite3.Connection):
    """Create tables and seed base_config."""
    conn.executescript("""
        CREATE TABLE base_config (
            key         TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            description TEXT
        );

        CREATE TABLE blocks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            domain          TEXT NOT NULL DEFAULT 'general',
            block_type      TEXT NOT NULL CHECK(block_type IN ('chat','task','technical')),
            content         TEXT NOT NULL,
            summary         TEXT,
            metadata_json   TEXT,
            initial_weight  REAL NOT NULL DEFAULT 1.0,
            current_weight  REAL NOT NULL DEFAULT 1.0,
            recall_count    INTEGER NOT NULL DEFAULT 0,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            last_recalled_at TEXT,
            archived        INTEGER NOT NULL DEFAULT 0
        );

        CREATE VIRTUAL TABLE blocks_fts USING fts5(
            content, summary, domain,
            content='blocks',
            content_rowid='id',
            tokenize='trigram'
        );

        CREATE TRIGGER blocks_ai AFTER INSERT ON blocks BEGIN
            INSERT INTO blocks_fts(rowid, content, summary, domain)
            VALUES (new.id, new.content, new.summary, new.domain);
        END;

        CREATE TRIGGER blocks_ad AFTER DELETE ON blocks BEGIN
            INSERT INTO blocks_fts(blocks_fts, rowid, content, summary, domain)
            VALUES ('delete', old.id, old.content, old.summary, old.domain);
        END;

        CREATE TRIGGER blocks_au AFTER UPDATE ON blocks BEGIN
            INSERT INTO blocks_fts(blocks_fts, rowid, content, summary, domain)
            VALUES ('delete', old.id, old.content, old.summary, old.domain);
            INSERT INTO blocks_fts(rowid, content, summary, domain)
            VALUES (new.id, new.content, new.summary, new.domain);
        END;
    """)

    for key, (value, desc) in BASE_CONFIG_DEFAULTS.items():
        conn.execute(
            "INSERT OR IGNORE INTO base_config(key, value, description) VALUES (?, ?, ?)",
            (key, str(value), desc),
        )
    conn.commit()


def _get_config(conn: sqlite3.Connection, key: str) -> str:
    row = conn.execute("SELECT value FROM base_config WHERE key = ?", (key,)).fetchone()
    if row is None:
        raise KeyError(f"Config key '{key}' not found in base_config")
    return row[0]


# ── Recall Algorithm ──────────────────────────────────
def _decay_weight(conn: sqlite3.Connection, block: sqlite3.Row) -> float:
    rate_key = DECAY_RATES.get(block["block_type"], "task_decay_rate")
    rate = float(_get_config(conn, rate_key))
    created = block["created_at"]
    if created:
        created_dt = datetime.fromisoformat(created)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        hours = (now - created_dt).total_seconds() / 3600.0
    else:
        hours = 0
    return max(block["current_weight"] * math.exp(-rate * hours), 0.0)


def _recency_score(block: sqlite3.Row) -> float:
    last = block["last_recalled_at"]
    if not last:
        return 0.5
    last_dt = datetime.fromisoformat(last)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    hours = (now - last_dt).total_seconds() / 3600.0
    return math.exp(-math.log(2) * hours / 24.0)


def _popularity_score(block: sqlite3.Row) -> float:
    count = block["recall_count"]
    if count == 0:
        return 0.0
    return min(math.log(count + 1) / math.log(101), 1.0)


def _compute_final_score(conn: sqlite3.Connection, block: sqlite3.Row, fts5_score: float) -> float:
    w_fts5 = float(_get_config(conn, "fts5_weight"))
    w_recency = float(_get_config(conn, "recency_weight"))
    w_pop = float(_get_config(conn, "popularity_weight"))
    return round(
        fts5_score * w_fts5
        + _recency_score(block) * w_recency
        + _popularity_score(block) * w_pop,
        6,
    )


def _update_recall_stats(conn: sqlite3.Connection, block_id: int):
    ceiling = float(_get_config(conn, "weight_ceiling"))
    boost = float(_get_config(conn, "recall_boost"))
    conn.execute(
        """UPDATE blocks
           SET recall_count = recall_count + 1,
               current_weight = MIN(current_weight + ?, ?),
               last_recalled_at = datetime('now')
           WHERE id = ?""",
        (boost, ceiling, block_id),
    )
    conn.commit()


# ── Dedup Helpers ─────────────────────────────────────
def _trigrams(s: str) -> list[str]:
    s = s.lower()
    return [s[i:i+3] for i in range(len(s)-2)]


def _bump_existing(conn: sqlite3.Connection, row: sqlite3.Row, reason: str) -> str:
    boost = float(_get_config(conn, "recall_boost") or 0.1)
    ceiling = float(_get_config(conn, "weight_ceiling") or 5.0)
    new_weight = min(row["current_weight"] + boost, ceiling)
    conn.execute(
        "UPDATE blocks SET current_weight=?, last_recalled_at=datetime('now') WHERE id=?",
        (new_weight, row["id"]),
    )
    conn.commit()
    return (
        f"⚡ Block #{row['id']} bumped ({reason} prevented)\n"
        f"   Existing: [{row['domain']}:{row['block_type']}] {row['summary'][:80]}\n"
        f"   Weight: {row['current_weight']:.2f} → {new_weight:.2f}\n"
        f"   Tip: archive_block(#{row['id']}) first if you need a fresh block."
    )


# ── MCP Server ────────────────────────────────────────
mcp = FastMCP(
    "block-memory",
    instructions="""
Block-based memory system with quantified coefficient recall.

Core concepts:
- Blocks are typed: chat (fast decay), task (medium decay), technical (slow decay)
- Recall uses FTS5 text matching + temporal decay + recency + popularity
- Weight ceiling prevents information cocoons
- Below-threshold matches are silently dropped

Workflow:
1. When you learn something important, call `create_block` to store it
2. When you need context, call `recall_blocks` with a query
3. Use `list_blocks` to see what's in a domain
4. Use `archive_block` to retire obsolete blocks
""",
)


@mcp.tool()
def create_block(
    domain: str,
    block_type: str,
    content: str,
    summary: str = "",
    metadata_json: str = "{}",
) -> str:
    """Create a new memory block.

    Auto-deduplication: if a block with a similar summary already exists in the
    same domain, the existing block is bumped instead of creating a duplicate.

    Args:
        domain: Domain label (e.g. 'project-notes', 'claude-usage')
        block_type: One of 'chat', 'task', 'technical'
        content: Full text content to store and index
        summary: One-line summary for clue cards (≤80 chars recommended)
        metadata_json: Optional JSON string with extended metadata
    """
    if block_type not in ("chat", "task", "technical"):
        return f"Error: block_type must be one of chat/task/technical, got '{block_type}'"

    try:
        json.loads(metadata_json)
    except json.JSONDecodeError:
        return "Error: metadata_json is not valid JSON"

    conn = get_db()
    try:
        # ── Dedup ──
        if summary.strip():
            try:
                exact = conn.execute(
                    "SELECT id, summary, current_weight, domain, block_type FROM blocks "
                    "WHERE domain=? AND archived=0 AND summary=? LIMIT 1",
                    (domain, summary.strip()),
                ).fetchone()
                if exact is not None:
                    return _bump_existing(conn, exact, "exact duplicate")

                candidates = conn.execute(
                    "SELECT id, summary, current_weight, domain, block_type FROM blocks "
                    "WHERE domain=? AND archived=0 AND summary != '' "
                    "ORDER BY created_at DESC LIMIT 30",
                    (domain,),
                ).fetchall()

                new_tokens = set(_trigrams(summary.strip().lower()))
                if new_tokens:
                    for row in candidates:
                        existing = set(_trigrams(row["summary"].lower()))
                        if not existing:
                            continue
                        overlap = len(existing & new_tokens) / max(len(existing | new_tokens), 1)
                        if overlap >= 0.4:
                            return _bump_existing(conn, row, f"near-duplicate ({overlap:.0%} overlap)")
            except Exception:
                pass

        # ── Create ──
        cursor = conn.execute(
            """INSERT INTO blocks (domain, block_type, content, summary, metadata_json)
               VALUES (?, ?, ?, ?, ?)""",
            (domain, block_type, content, summary, metadata_json),
        )
        block_id = cursor.lastrowid
        conn.commit()

        block = conn.execute("SELECT * FROM blocks WHERE id = ?", (block_id,)).fetchone()
        return (
            f"✅ Block #{block['id']} created\n"
            f"   Domain: {block['domain']}\n"
            f"   Type: {block['block_type']} (decay: {DECAY_RATES[block['block_type']]})\n"
            f"   Initial weight: {block['initial_weight']}\n"
            f"   Content preview: {content[:100]}..."
        )
    finally:
        conn.close()


@mcp.tool()
def recall_blocks(query: str, domain: str = "", max_results: int = 10) -> str:
    """Search memory blocks using quantified coefficient recall.

    Combines FTS5 text match + temporal decay + recency + popularity.
    Below-threshold results are silently dropped.

    Args:
        query: Search query (natural language)
        domain: Optional domain filter (empty = all domains)
        max_results: Maximum blocks to return (default 10)
    """
    conn = get_db()
    threshold = float(_get_config(conn, "recall_threshold"))

    try:
        # FTS5 search
        if domain:
            fts5_rows = conn.execute(
                """SELECT b.*, f.rank AS fts5_rank
                   FROM blocks_fts f JOIN blocks b ON b.id = f.rowid
                   WHERE blocks_fts MATCH ? AND b.archived = 0 AND b.domain = ?
                   ORDER BY f.rank LIMIT ?""",
                (query, domain, max_results * 3),
            ).fetchall()
        else:
            fts5_rows = conn.execute(
                """SELECT b.*, f.rank AS fts5_rank
                   FROM blocks_fts f JOIN blocks b ON b.id = f.rowid
                   WHERE blocks_fts MATCH ? AND b.archived = 0
                   ORDER BY f.rank LIMIT ?""",
                (query, max_results * 3),
            ).fetchall()

        # LIKE fallback
        like_query = f"%{query}%"
        if domain:
            like_rows = conn.execute(
                """SELECT b.*, -3.0 AS fts5_rank FROM blocks b
                   WHERE (b.content LIKE ? OR b.summary LIKE ?) AND b.archived = 0 AND b.domain = ?
                   LIMIT ?""",
                (like_query, like_query, domain, max_results * 2),
            ).fetchall()
        else:
            like_rows = conn.execute(
                """SELECT b.*, -3.0 AS fts5_rank FROM blocks b
                   WHERE (b.content LIKE ? OR b.summary LIKE ?) AND b.archived = 0
                   LIMIT ?""",
                (like_query, like_query, max_results * 2),
            ).fetchall()

        # Merge: FTS5 preferred, LIKE as fallback
        merged = {}
        for row in fts5_rows:
            merged[row["id"]] = row
        for row in like_rows:
            if row["id"] not in merged:
                merged[row["id"]] = row

        if not merged:
            return "🔍 No matching blocks found."

        rows = list(merged.values())
        NORM_DIVISOR = 6.0

        scored = []
        for row in rows:
            fts5_raw = row["fts5_rank"] if row["fts5_rank"] is not None else -3.0
            fts5_score = min(1.0, abs(fts5_raw) / NORM_DIVISOR) if fts5_raw < 0 else 0.3

            decayed = _decay_weight(conn, row)
            if decayed < threshold:
                continue

            final = _compute_final_score(conn, row, fts5_score)
            if final < threshold:
                continue

            scored.append((row, final, decayed, fts5_score))

        if not scored:
            return "🔍 No matching blocks passed the threshold filter."

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:max_results]

        ceiling = float(_get_config(conn, "weight_ceiling"))
        lines = [f"## Memory Recall: \"{query}\"\n"]
        lines.append("| # | Score | Decay | FTS5 | ID | Domain | Type | Content |")
        lines.append("|---|------:|------:|-----:|---:|--------|------|--------|")

        for i, (block, final, decayed, fts5) in enumerate(scored, 1):
            preview = block["content"][:80].replace("\n", " ").replace("|", "\\|")
            lines.append(
                f"| {i} | {final:.4f} | {decayed:.4f} | {fts5:.4f} | "
                f"{block['id']} | {block['domain']} | {block['block_type']} | {preview} |"
            )
            _update_recall_stats(conn, block["id"])

        lines.append(f"\n**{len(scored)} blocks recalled** | Threshold: {threshold} | Ceiling: {ceiling}")
        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
def get_block(block_id: int) -> str:
    """Retrieve a single block by ID. Bumps recall_count and weight.

    Args:
        block_id: The block ID to retrieve
    """
    conn = get_db()
    try:
        block = conn.execute("SELECT * FROM blocks WHERE id = ?", (block_id,)).fetchone()
        if not block:
            return f"Error: Block #{block_id} not found."

        _update_recall_stats(conn, block_id)
        block = conn.execute("SELECT * FROM blocks WHERE id = ?", (block_id,)).fetchone()
        decayed = _decay_weight(conn, block)

        lines = [
            f"## Block #{block['id']}",
            f"- **Domain:** {block['domain']}",
            f"- **Type:** {block['block_type']}",
            f"- **Created:** {block['created_at']}",
            f"- **Last recalled:** {block['last_recalled_at'] or 'never'}",
            f"- **Current weight:** {block['current_weight']:.4f} (decayed: {decayed:.4f})",
            f"- **Recall count:** {block['recall_count']}",
            f"- **Archived:** {'yes' if block['archived'] else 'no'}",
        ]
        if block["summary"]:
            lines.insert(7, f"- **Summary:** {block['summary']}")
        lines.append("\n### Content")
        lines.append(block["content"])
        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
def list_blocks(domain: str = "", block_type: str = "", limit: int = 20) -> str:
    """List memory blocks, sorted by current weight descending.

    Args:
        domain: Optional domain filter
        block_type: Optional type filter (chat/task/technical)
        limit: Max blocks to return (default 20)
    """
    conn = get_db()
    try:
        conditions = ["archived = 0"]
        params = []
        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if block_type:
            conditions.append("block_type = ?")
            params.append(block_type)

        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM blocks WHERE {where} ORDER BY current_weight DESC LIMIT ?",
            (*params, limit),
        ).fetchall()

        if not rows:
            return "No blocks found."

        lines = [
            f"## Memory Blocks ({len(rows)} shown)\n",
            "| ID | Domain | Type | Weight | Recalls | Created | Preview |",
            "|----|--------|------|-------:|--------:|---------|---------|",
        ]
        for b in rows:
            preview = b["content"][:60].replace("\n", " ").replace("|", "\\|")
            lines.append(
                f"| {b['id']} | {b['domain']} | {b['block_type']} | "
                f"{b['current_weight']:.3f} | {b['recall_count']} | "
                f"{b['created_at'][:10]} | {preview} |"
            )
        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
def archive_block(block_id: int) -> str:
    """Archive a block (cold storage, excluded from recall).

    Args:
        block_id: The block ID to archive
    """
    conn = get_db()
    try:
        block = conn.execute("SELECT * FROM blocks WHERE id = ?", (block_id,)).fetchone()
        if not block:
            return f"Error: Block #{block_id} not found."
        conn.execute("UPDATE blocks SET archived = 1 WHERE id = ?", (block_id,))
        conn.commit()
        return f"📦 Block #{block_id} archived. Will no longer appear in recall results."
    finally:
        conn.close()


# ── Entry Point ───────────────────────────────────────
def main():
    """CLI entry point: start the MCP server on stdio transport."""
    get_db().close()  # ensure DB exists before serving
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
