"""Test orchestration for running all LLM tests."""

from pathlib import Path

from ibm_watsonx_ai import APIClient

from modelgrader.gemini_grader import grade_response
from modelgrader.logging import get_logger
from modelgrader.models import Question, TestResult
from modelgrader.watsonx_client import create_prompt, query_model

logger = get_logger(__name__)


def load_questions(questions_dir: str | Path, contexts_dir: str | Path) -> list[Question]:
    """Load questions and their corresponding contexts.

    Args:
        questions_dir: Directory containing question files
        contexts_dir: Directory containing context files

    Returns:
        List of Question objects
    """
    questions_path = Path(questions_dir)
    contexts_path = Path(contexts_dir)

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions directory not found: {questions_path}")

    if not contexts_path.exists():
        raise FileNotFoundError(f"Contexts directory not found: {contexts_path}")

    # Get all question files (sorted by number)
    question_files = sorted(questions_path.glob("question_*.txt"))

    questions = []
    for question_file in question_files:
        # Extract question number from filename (e.g., "question_1.txt" -> 1)
        number = int(question_file.stem.split("_")[1])

        # Read question text
        question_text = question_file.read_text(encoding="utf-8").strip()

        # Find corresponding context file
        context_file = contexts_path / f"context_{number}.txt"

        questions.append(
            Question(
                number=number,
                text=question_text,
                context_path=context_file if context_file.exists() else None,
            )
        )

    logger.info("questions_loaded", count=len(questions))
    return questions


def run_single_test(
    client: APIClient,
    model_id: str,
    question: Question,
    with_context: bool,
) -> TestResult:
    """Run a single test of a model with a question.

    Args:
        client: WatsonX API client
        model_id: Model ID to test
        question: Question to ask
        with_context: Whether to include context

    Returns:
        TestResult with grades
    """
    context = question.load_context() if with_context else None
    prompt = create_prompt(question.text, context)

    logger.info(
        "running_test",
        model_id=model_id,
        question_number=question.number,
        with_context=with_context,
    )

    # Query the model
    response, response_time = query_model(client, model_id, prompt)

    # Grade the response
    grades = grade_response(
        question=question.text,
        response=response,
        context=context,
        response_time=response_time,
    )

    # Create result
    result = TestResult(
        model_name=model_id,
        question_number=question.number,
        question_text=question.text,
        context_provided=with_context,
        response=response,
        response_time=response_time,
        grades=grades,
    )

    logger.info(
        "test_complete",
        model_id=model_id,
        question_number=question.number,
        with_context=with_context,
        total_score=result.total_score,
    )

    return result


def run_all_tests(
    client: APIClient,
    model_ids: list[str],
    questions: list[Question],
) -> list[TestResult]:
    """Run all tests for all models and questions.

    Args:
        client: WatsonX API client
        model_ids: List of model IDs to test
        questions: List of questions to ask

    Returns:
        List of all test results
    """
    results = []
    total_tests = len(model_ids) * len(questions) * 2  # x2 for with/without context

    logger.info(
        "starting_all_tests",
        model_count=len(model_ids),
        question_count=len(questions),
        total_tests=total_tests,
    )

    for model_id in model_ids:
        for question in questions:
            # Test without context
            try:
                result = run_single_test(
                    client=client,
                    model_id=model_id,
                    question=question,
                    with_context=False,
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    "test_failed",
                    model_id=model_id,
                    question_number=question.number,
                    with_context=False,
                    error=str(e),
                )

            # Test with context
            try:
                result = run_single_test(
                    client=client,
                    model_id=model_id,
                    question=question,
                    with_context=True,
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    "test_failed",
                    model_id=model_id,
                    question_number=question.number,
                    with_context=True,
                    error=str(e),
                )

    logger.info("all_tests_complete", results_count=len(results))
    return results
