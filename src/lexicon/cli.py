from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from lexicon.index import LexiconIndex, make_embedder
from lexicon.reconcile import appears_contested, reconcile as reconcile_passages

app = typer.Typer(help="Lexicon: local-first semantic pointer search over documents.")
console = Console()


def _default_index_path() -> Path:
    return Path(".lexicon/index.db")


@app.command()
def index(
    paths: Annotated[list[Path], typer.Argument(help="Files or directories to index.")],
    index_path: Annotated[Path, typer.Option("--index", help="SQLite index path.")] = _default_index_path(),
    backend: Annotated[
        str,
        typer.Option("--backend", help="Embedding backend: auto, hashing, or sentence-transformers."),
    ] = "auto",
    model: Annotated[
        str | None,
        typer.Option("--model", help="SentenceTransformer model name when using that backend."),
    ] = None,
) -> None:
    """Build a local passage index from documents."""

    embedder = make_embedder(backend=backend, model_name=model)
    lexicon = LexiconIndex(index_path)
    try:
        count = lexicon.rebuild(paths, embedder)
    finally:
        lexicon.close()

    console.print(f"Indexed {count} passages into [bold]{index_path}[/bold] using {embedder.name}.")


@app.command()
def ask(
    query: Annotated[str, typer.Argument(help="Question to point at source passages for.")],
    index_path: Annotated[Path, typer.Option("--index", help="SQLite index path.")] = _default_index_path(),
    backend: Annotated[
        str,
        typer.Option("--backend", help="Embedding backend used for the query vector."),
    ] = "auto",
    model: Annotated[
        str | None,
        typer.Option("--model", help="SentenceTransformer model name when using that backend."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-k", min=1, max=25)] = 5,
    reconcile: Annotated[
        bool,
        typer.Option("--reconcile", help="Make one reconciliation call over retrieved passages."),
    ] = False,
    auto_reconcile: Annotated[
        bool,
        typer.Option("--auto-reconcile", help="Reconcile only if retrieved passages appear contested."),
    ] = False,
) -> None:
    """Return cited passages. This command does not call an LLM unless asked to reconcile."""

    embedder = make_embedder(backend=backend, model_name=model)
    lexicon = LexiconIndex(index_path)
    try:
        try:
            results = lexicon.search(query, embedder=embedder, limit=limit)
        except ValueError as error:
            console.print(f"[red]{escape(str(error))}[/red]")
            raise typer.Exit(code=2) from error
    finally:
        lexicon.close()

    if not results:
        console.print("No passages found. Run [bold]lexicon index <path>[/bold] first.")
        raise typer.Exit(code=1)

    _print_results(results)

    contested = appears_contested(results)
    if auto_reconcile and contested:
        console.print("\n[bold yellow]Potential disagreement detected; reconciling cited passages.[/bold yellow]")
        console.print(Panel(reconcile_passages(query, results).explanation, title="Reconciliation"))
    elif reconcile:
        console.print("\n[bold yellow]Explicit reconciliation requested.[/bold yellow]")
        console.print(Panel(reconcile_passages(query, results).explanation, title="Reconciliation"))
    elif contested:
        console.print(
            "\n[yellow]Potential disagreement cues found. Re-run with --reconcile or --auto-reconcile "
            "to invoke the configured reconciliation layer.[/yellow]"
        )


def _print_results(results) -> None:
    table = Table(title="Relevant Source Passages", show_lines=True)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Citation", style="cyan", no_wrap=False)
    table.add_column("Passage", no_wrap=False)

    for result in results:
        table.add_row(
            f"{result.score:.3f}",
            escape(result.passage.citation),
            escape(result.passage.text),
        )

    console.print(table)


if __name__ == "__main__":
    app()

