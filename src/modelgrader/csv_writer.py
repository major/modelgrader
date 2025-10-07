"""CSV output writer for test results."""

import csv
from pathlib import Path

from modelgrader.logging import get_logger
from modelgrader.models import GradeBreakdown, TestResult

logger = get_logger(__name__)

# CSV field names (constant for consistency)
CSV_FIELDNAMES = [
    "Model Name",
    "Question",
    "Context Provided",
    "Accuracy Score",
    "Completeness Score",
    "Clarity Score",
    "Response time",
    "Weighted Score",
    "Percentile Rank",
    "Explanation",
]


def initialize_csv(output_path: str | Path) -> None:
    """Initialize a CSV file with headers if it doesn't exist.

    Args:
        output_path: Path to output CSV file
    """
    output_path = Path(output_path)

    if not output_path.exists():
        logger.info("initializing_csv", path=str(output_path))
        try:
            with output_path.open("w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()
            logger.info("csv_initialized", path=str(output_path))
        except Exception as e:
            logger.error("csv_init_failed", path=str(output_path), error=str(e))
            raise


def append_result_to_csv(result: TestResult, output_path: str | Path) -> None:
    """Append a single test result to the CSV file.

    Args:
        result: Test result to append
        output_path: Path to output CSV file
    """
    output_path = Path(output_path)

    try:
        with output_path.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
            row = result.to_csv_row()
            writer.writerow(row)

        logger.debug("result_appended_to_csv", model=result.model_name, question=result.question_number)

    except Exception as e:
        logger.error("csv_append_failed", path=str(output_path), error=str(e))
        raise


def load_existing_results(output_path: str | Path) -> set[tuple[str, int, bool]]:
    """Load existing test results from CSV to determine what's already been tested.

    Args:
        output_path: Path to CSV file

    Returns:
        Set of (model_name, question_number, context_provided) tuples
    """
    output_path = Path(output_path)

    if not output_path.exists():
        logger.info("no_existing_csv", path=str(output_path))
        return set()

    existing = set()

    try:
        with output_path.open("r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                model_name = row["Model Name"]
                # Extract question number from "Q1", "Q2", etc.
                question_num = int(row["Question"].replace("Q", ""))
                context_provided = row["Context Provided"] == "Yes"
                existing.add((model_name, question_num, context_provided))

        logger.info("loaded_existing_results", count=len(existing), path=str(output_path))
        return existing

    except Exception as e:
        logger.error("failed_to_load_existing_results", path=str(output_path), error=str(e))
        # If we can't load existing results, return empty set to start fresh
        return set()


def load_all_results(output_path: str | Path) -> list[TestResult]:
    """Load all test results from CSV file.

    Args:
        output_path: Path to CSV file

    Returns:
        List of TestResult objects
    """
    output_path = Path(output_path)

    if not output_path.exists():
        return []

    results = []

    try:
        with output_path.open("r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Calculate response time score based on response time
                response_time = float(row["Response time"])
                if response_time <= 10:
                    response_time_score = 95
                elif response_time <= 30:
                    response_time_score = 80
                elif response_time <= 60:
                    response_time_score = 60
                elif response_time <= 90:
                    response_time_score = 40
                elif response_time <= 120:
                    response_time_score = 20
                else:
                    response_time_score = 5

                # Reconstruct TestResult from CSV row
                result = TestResult(
                    model_name=row["Model Name"],
                    question_number=int(row["Question"].replace("Q", "")),
                    question_text="",  # Not stored in CSV
                    context_provided=row["Context Provided"] == "Yes",
                    response="",  # Not stored in CSV
                    response_time=response_time,
                    grades=GradeBreakdown(
                        accuracy=int(row["Accuracy Score"]),
                        completeness=int(row["Completeness Score"]),
                        clarity=int(row["Clarity Score"]),
                        response_time_score=response_time_score,
                    ),
                    percentile=float(row.get("Percentile Rank", 0.0)),
                )
                results.append(result)

        logger.info("loaded_all_results", count=len(results), path=str(output_path))
        return results

    except Exception as e:
        logger.error("failed_to_load_all_results", path=str(output_path), error=str(e))
        return []


def write_results_to_csv(results: list[TestResult], output_path: str | Path) -> None:
    """Write all test results to a CSV file (overwrites existing file).

    This is used for the final write with percentiles calculated.

    Args:
        results: List of test results
        output_path: Path to output CSV file
    """
    if not results:
        logger.warning("no_results_to_write")
        return

    output_path = Path(output_path)

    logger.info("writing_csv", path=str(output_path), result_count=len(results))

    try:
        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()

            for result in results:
                row = result.to_csv_row()
                writer.writerow(row)

        logger.info("csv_written_successfully", path=str(output_path))

    except Exception as e:
        logger.error("csv_write_failed", path=str(output_path), error=str(e))
        raise
