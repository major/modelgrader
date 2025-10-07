"""Rich console output for displaying test results."""

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from modelgrader.models import TestResult

console = Console()


def create_progress_bar() -> Progress:
    """Create a configured progress bar for test execution.

    Returns:
        Configured Progress instance
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    )


def print_header() -> None:
    """Print the application header."""
    console.print(
        Panel.fit(
            "[bold cyan]LLM Grading System[/bold cyan]\n"
            "[dim]Testing WatsonX models with RHEL questions[/dim]",
            border_style="cyan",
        )
    )
    console.print()


def print_models_info(model_count: int, models: list[str]) -> None:
    """Print information about discovered models.

    Args:
        model_count: Number of models found
        models: List of model IDs
    """
    console.print(f"[bold green]✓[/bold green] Found {model_count} text/chat models")

    # Show first few models
    if models:
        sample = models[:5]
        console.print(f"[dim]  Sample: {', '.join(sample)}{'...' if len(models) > 5 else ''}[/dim]")
    console.print()


def print_questions_info(question_count: int) -> None:
    """Print information about loaded questions.

    Args:
        question_count: Number of questions loaded
    """
    console.print(
        f"[bold green]✓[/bold green] Loaded {question_count} questions with contexts"
    )
    console.print()


def print_resume_info(existing_count: int, total_count: int) -> None:
    """Print information about resuming from existing results.

    Args:
        existing_count: Number of existing test results found
        total_count: Total number of tests to run
    """
    if existing_count > 0:
        remaining = total_count - existing_count
        console.print(
            f"[bold yellow]↻[/bold yellow] Found {existing_count} existing results, "
            f"{remaining} tests remaining"
        )
        console.print()


def create_results_table() -> Table:
    """Create a table for displaying results.

    Returns:
        Configured Table instance
    """
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Model", style="cyan", no_wrap=False, width=25)
    table.add_column("Q#", justify="center", width=4)
    table.add_column("Ctx", justify="center", width=4)
    table.add_column("Score", justify="right", width=7)
    table.add_column("%ile", justify="right", width=6)
    table.add_column("Acc", justify="right", width=5)
    table.add_column("Com", justify="right", width=5)
    table.add_column("Clr", justify="right", width=5)
    table.add_column("Time", justify="right", width=6)
    return table


def add_result_to_table(table: Table, result: TestResult) -> None:
    """Add a test result to the table.

    Args:
        table: Table to add result to
        result: Test result to add
    """
    # Determine color based on percentile ranking
    percentile = result.percentile
    if percentile >= 75:
        percentile_color = "green"
    elif percentile >= 50:
        percentile_color = "yellow"
    elif percentile >= 25:
        percentile_color = "blue"
    else:
        percentile_color = "red"

    # Truncate model name if too long
    model_display = result.model_name
    if len(model_display) > 25:
        model_display = model_display[:22] + "..."

    table.add_row(
        model_display,
        str(result.question_number),
        "Y" if result.context_provided else "N",
        f"{result.total_score:.1f}",
        f"[{percentile_color}]{percentile:.1f}[/{percentile_color}]",
        str(result.grades.accuracy),
        str(result.grades.completeness),
        str(result.grades.clarity),
        f"{result.response_time:.1f}s",
    )


def print_results_table(table: Table) -> None:
    """Print the results table.

    Args:
        table: Table to print
    """
    console.print()
    console.print(table)
    console.print()


def print_summary(results: list[TestResult]) -> None:
    """Print summary statistics.

    Args:
        results: List of all test results
    """
    if not results:
        console.print("[yellow]No results to summarize[/yellow]")
        return

    total_tests = len(results)
    avg_score = sum(r.total_score for r in results) / total_tests
    avg_time = sum(r.response_time for r in results) / total_tests

    # Find best and worst
    best = max(results, key=lambda r: r.total_score)
    worst = min(results, key=lambda r: r.total_score)

    # Calculate with/without context comparison
    with_context = [r for r in results if r.context_provided]
    without_context = [r for r in results if not r.context_provided]

    avg_with_context = (
        sum(r.total_score for r in with_context) / len(with_context)
        if with_context
        else 0
    )
    avg_without_context = (
        sum(r.total_score for r in without_context) / len(without_context)
        if without_context
        else 0
    )

    summary_text = f"""[bold]Test Summary[/bold]

Total Tests: {total_tests}
Average Weighted Score: {avg_score:.1f}/100
Average Response Time: {avg_time:.1f}s

[bold]Best Performance:[/bold]
  Model: {best.model_name}
  Question: {best.question_number}
  Score: {best.total_score:.1f}/100 (Percentile: {best.percentile:.1f})

[bold]Worst Performance:[/bold]
  Model: {worst.model_name}
  Question: {worst.question_number}
  Score: {worst.total_score:.1f}/100 (Percentile: {worst.percentile:.1f})

[bold]Context Impact:[/bold]
  With Context: {avg_with_context:.1f}/100
  Without Context: {avg_without_context:.1f}/100
  Improvement: {avg_with_context - avg_without_context:+.1f} points"""

    console.print(Panel(summary_text, border_style="green", title="Summary"))
    console.print()


def print_error(message: str) -> None:
    """Print an error message.

    Args:
        message: Error message to display
    """
    console.print(f"[bold red]✗ Error:[/bold red] {message}")


def print_success(message: str) -> None:
    """Print a success message.

    Args:
        message: Success message to display
    """
    console.print(f"[bold green]✓[/bold green] {message}")
