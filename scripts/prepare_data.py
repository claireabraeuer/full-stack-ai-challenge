#!/usr/bin/env python
"""CLI script to run the data preparation pipeline."""

import sys
from pathlib import Path

import typer
from rich.console import Console

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pipeline import DataPipeline

app = typer.Typer(help="Data preparation pipeline for support tickets")
console = Console()


@app.command()
def prepare(
    data_path: str = typer.Option(
        "support_tickets.json",
        "--data",
        "-d",
        help="Path to input JSON file",
    ),
    output_dir: str = typer.Option(
        "data/splits",
        "--output",
        "-o",
        help="Output directory for splits",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip data quality validation",
    ),
) -> None:
    """Run the data preparation pipeline."""
    try:
        console.print(f"\n[bold blue]Data Preparation Pipeline[/bold blue]")
        console.print(f"Input:  {data_path}")
        console.print(f"Output: {output_dir}\n")

        pipeline = DataPipeline(
            data_path=data_path,
            output_dir=output_dir,
            validate_quality=not skip_validation,
        )

        train_df, val_df, test_df = pipeline.run()

        console.print("\n[bold green]✓ Pipeline completed successfully![/bold green]")
        console.print(f"\nGenerated splits:")
        console.print(f"  • Train: {len(train_df):,} samples")
        console.print(f"  • Val:   {len(val_df):,} samples")
        console.print(f"  • Test:  {len(test_df):,} samples")

    except Exception as e:
        console.print(f"\n[bold red]✗ Pipeline failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def verify(
    splits_dir: str = typer.Option(
        "data/splits",
        "--dir",
        "-d",
        help="Directory containing splits",
    ),
) -> None:
    """Verify that data splits can be loaded."""
    from src.data.splitter import load_splits

    try:
        console.print(f"\n[bold blue]Verifying splits in {splits_dir}...[/bold blue]\n")

        train_df, val_df, test_df = load_splits(Path(splits_dir))

        console.print("[bold green]✓ All splits loaded successfully![/bold green]")
        console.print(f"\nShapes:")
        console.print(f"  • Train: {train_df.shape}")
        console.print(f"  • Val:   {val_df.shape}")
        console.print(f"  • Test:  {test_df.shape}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Verification failed: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
