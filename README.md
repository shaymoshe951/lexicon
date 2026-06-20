# Lexicon

Lexicon is a local-first document knowledge tool. It indexes local documents as rich passages, then answers questions by returning the most relevant source passages with citations. It is a pointer engine, not a synthesizer: the default query path does not call an LLM and does not invent an answer.

On top of retrieval there is a thin reconciliation layer. It runs only when you explicitly pass `--reconcile`, or when you opt into `--auto-reconcile` and retrieved passages appear to disagree. Reconciliation receives only the retrieved passages and must preserve divergent positions with their sources.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For stronger local semantic search, install the optional transformer backend:

```bash
pip install -e ".[semantic,dev]"
```

## Use

Index documents:

```bash
lexicon index ~/Documents/notes
```

Ask for source passages:

```bash
lexicon ask "What does the policy say about remote work?"
```

Use a local transformer model:

```bash
lexicon index ~/Documents/notes --backend sentence-transformers
lexicon ask "What does the policy say about remote work?" --backend sentence-transformers
```

Reconcile only the retrieved passages:

```bash
export LEXICON_LLM_COMMAND="your-local-llm-command"
lexicon ask "Is remote work allowed?" --reconcile
```

`LEXICON_LLM_COMMAND` must be a command that reads one JSON payload from stdin and writes the reconciliation text to stdout. Lexicon sends the query plus retrieved passages only.

## Supported Documents

The first version supports plain local text formats: `.txt`, `.md`, `.rst`, `.csv`, `.json`, `.log`, and `.text`.

## Design Rules

- Keep source passages rich enough to preserve context and scope.
- Return filename and location for every passage.
- Do not synthesize an answer on the default path.
- Do not build fact extractors, claim stores, or a knowledge graph.
- Never silently resolve a conflict; keep and show both sides.

