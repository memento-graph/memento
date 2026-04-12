---
name: memento-memory
description: "Bitemporal knowledge graph memory — captures conversations, extracts entities and relationships via LLM, detects contradictions, and recalls relevant context using semantic + keyword + graph search. Any model, same memory."
homepage: https://github.com/shane-farkas/memento-memory
user-invocable: true
metadata: {"openclaw": {"emoji": "🧠", "os": ["darwin", "linux", "win32"], "requires": {"anyBins": ["memento-mcp"]}, "primaryEnv": "ANTHROPIC_API_KEY", "install": [{"id": "uv-anthropic", "kind": "uv", "package": "memento-memory[anthropic]", "bins": ["memento-mcp"], "label": "Install with Anthropic/Claude (recommended)"}, {"id": "uv-openai", "kind": "uv", "package": "memento-memory[openai]", "bins": ["memento-mcp"], "label": "Install with OpenAI"}, {"id": "uv-gemini", "kind": "uv", "package": "memento-memory[gemini]", "bins": ["memento-mcp"], "label": "Install with Google Gemini"}, {"id": "uv-ollama", "kind": "uv", "package": "memento-memory[openai]", "bins": ["memento-mcp"], "label": "Install with Ollama (local, no API key)"}]}}
---

# Memento — Persistent Memory

Memento is your long-term memory. It builds a **bitemporal knowledge graph** from conversations — extracting entities, resolving duplicates, detecting contradictions, and tracking how facts change over time. Everything is stored locally in SQLite.

## When to Use Each Tool

### memory_ingest — Store information

Call `memory_ingest` to remember important facts from conversations. Use it when:

- The user shares personal details, preferences, or decisions
- New facts are mentioned about people, projects, organizations, or events
- Information changes or updates (Memento will detect contradictions automatically)
- The user explicitly asks you to remember something

```
memory_ingest(text="User prefers dark mode. Works at Acme Corp as a senior engineer.")
```

You do NOT need to ingest every message. Focus on facts worth remembering — preferences, decisions, relationships, key events.

### memory_recall — Retrieve relevant context

Call `memory_recall` **before answering questions** that might benefit from past context. Use it when:

- The user asks about something you may have discussed before
- The user references a person, project, or topic from past conversations
- You need context about the user's preferences or situation
- The user asks "do you remember" or "what do you know about"

```
memory_recall(query="What do I know about John's project?", token_budget=2000)
```

The system returns a composed briefing from the knowledge graph — not just raw text chunks, but structured information with entities, relationships, and relevant conversation excerpts.

### memory_recall_as_of — Point-in-time queries

When the user asks about the past state of something:

```
memory_recall_as_of(query="John's job title", as_of="2025-01-31T00:00:00Z")
```

This returns what was known at that specific point in time — useful for "what was X before it changed?" questions.

### memory_entities — Browse the knowledge graph

List all known entities, optionally filtered by type:

```
memory_entities(type_filter="person")
```

Types: person, organization, project, location, concept, event.

### memory_entity — Deep dive on one entity

Get full details including properties, relationships, and confidence scores:

```
memory_entity(entity_id="...")
```

### memory_correct — Fix wrong information

When the user says something is wrong in memory:

```
memory_correct(entity_id="...", property_key="title", new_value="CTO", reason="User corrected")
```

### memory_forget — Remove information

When the user asks you to forget something:

```
memory_forget(entity_id="...")
```

This is a soft delete — the entity is archived, not destroyed.

### memory_merge — Deduplicate entities

When you notice the same real-world thing has two entries:

```
memory_merge(entity_a_id="...", entity_b_id="...")
```

### memory_conflicts — Check contradictions

Review unresolved contradictions in the knowledge graph:

```
memory_conflicts()
```

### memory_health — System status

Check the state of the knowledge graph:

```
memory_health()
```

Returns entity count, relationship count, property count, average confidence, and unresolved conflicts.

## How It Works

1. **Ingestion**: Text goes through entity extraction (LLM), entity resolution (fuzzy/phonetic/embedding matching), relationship extraction, contradiction detection, and verbatim storage
2. **Retrieval**: Queries search via FTS5 keywords, semantic embeddings, and knowledge graph traversal — then results are ranked and assembled within a token budget
3. **Temporal tracking**: Every fact records when it was true in the world (valid time) and when the system learned it (transaction time)
4. **Consolidation**: Background engine decays stale information, merges duplicates, and prunes orphans

## Best Practices

- **Recall before responding** when the conversation might involve previously discussed topics
- **Ingest selectively** — facts, preferences, and decisions, not filler
- **Use as_of for temporal queries** — "what was true in January?" needs a timestamp
- **Don't ingest the same text twice** — Memento handles dedup but it wastes LLM calls
- **Let the user know** what you remembered or recalled when it's relevant to the conversation

## LLM Provider Configuration

Memento works with any LLM backend. Set one of these environment variables:

- `ANTHROPIC_API_KEY` — for Claude (default)
- `OPENAI_API_KEY` with `MEMENTO_LLM_PROVIDER=openai` — for OpenAI
- `GOOGLE_API_KEY` with `MEMENTO_LLM_PROVIDER=gemini` — for Gemini
- `MEMENTO_LLM_PROVIDER=ollama` — for fully local inference via Ollama

All data stays local in `~/.memento/memento.db`.
