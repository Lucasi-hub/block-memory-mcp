# Block Memory MCP Server

区块化动态系数召回AI记忆 — an MCP server that gives LLMs persistent, searchable memory with automatic decay.

## Why

LLMs forget between sessions. Block Memory gives them a persistent memory system:
- **Store** facts, decisions, and project context as typed blocks
- **Search** with FTS5 full-text search across all blocks
- **Auto-decay** — irrelevant memories fade over time (chat: fast, technical: slow)
- **Auto-dedup** — creating a similar block bumps the existing one instead
- **Weight ceiling** — prevents popular blocks from drowning out everything else

## Quick Start

```bash
pip install git+https://github.com/Lucasi-hub/block-memory-mcp.git
```

## Configuration

### Claude Code

Add to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "block-memory": {
      "type": "stdio",
      "command": "python3",
      "args": ["-m", "block_memory"]
    }
  }
}
```

### Claude Desktop
{
    "mcpServers": {
      "block-memory": {
        "type": "stdio",
        "command": "python3",
        "args": ["-m", "block_memory"],
        "env": {}
      }
    }
}

### Custom Database Path

```bash
export BLOCK_MEMORY_DB=~/my-project/memory.db
```

Default: `~/.block-memory/blocks.db`

## Tools

| Tool | Description |
|------|-------------|
| `create_block` | Store a new memory. Auto-dedup on summary. |
| `recall_blocks` | Search with FTS5 + temporal decay + recency + popularity |
| `get_block` | Retrieve one block by ID. Bumps weight. |
| `list_blocks` | Browse blocks by domain/type, sorted by weight |
| `archive_block` | Move to cold storage (excluded from recall) |

## Block Types

| Type | Decay Rate | Half-life | Use For |
|------|-----------|-----------|---------|
| `chat` | 5%/hour | ~14 hours | Transient conversation context |
| `task` | 2%/hour | ~35 hours | Task outcomes, session summaries |
| `technical` | 0.5%/hour | ~139 hours | Design decisions, discoveries, persistent facts |

## How Recall Works

```
Final Score = FTS5_match (0.6) + Recency (0.2) + Popularity (0.2)
```

- **FTS5**: Trigram tokenizer handles Chinese + English. LIKE fallback catches 2-char queries.
- **Temporal decay**: Older blocks fade exponentially based on their type.
- **Recency**: Blocks recalled recently get a boost.
- **Popularity**: Frequently-recalled blocks rank higher (log-scaled).

Blocks scoring below the threshold (0.1) are silently dropped to reduce noise.

## License

MIT
