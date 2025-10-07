"""Shared pytest fixtures for the test suite."""

from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pytest

from modelgrader.models import GradeBreakdown, Question, TestResult


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_csv_file(temp_dir):
    """Create a temporary CSV file path for testing."""
    # Return a path that doesn't exist yet
    csv_path = temp_dir / "test_results.csv"
    yield csv_path
    # Cleanup
    csv_path.unlink(missing_ok=True)


@pytest.fixture
def sample_question(temp_dir):
    """Create a sample question with context file."""
    context_file = temp_dir / "context_1.txt"
    context_file.write_text("This is sample context for SELinux troubleshooting.")

    return Question(
        number=1,
        text="How do I troubleshoot SELinux denials?",
        context_path=context_file,
    )


@pytest.fixture
def sample_grade_breakdown():
    """Create a sample grade breakdown."""
    return GradeBreakdown(
        accuracy=85,
        completeness=75,
        clarity=80,
        response_time_score=90,
    )


@pytest.fixture
def sample_test_result(sample_grade_breakdown):
    """Create a sample test result."""
    return TestResult(
        model_name="ibm/granite-3.1-8b-instruct",
        question_number=1,
        question_text="How do I troubleshoot SELinux denials?",
        context_provided=True,
        response="To troubleshoot SELinux denials, check /var/log/audit/audit.log...",
        response_time=5.2,
        grades=sample_grade_breakdown,
        percentile=75.5,
    )


@pytest.fixture
def multiple_test_results():
    """Create multiple test results for testing aggregations."""
    results = []

    # Create 10 test results with varying scores
    for i in range(10):
        grades = GradeBreakdown(
            accuracy=70 + i * 2,
            completeness=60 + i * 3,
            clarity=65 + i * 2,
            response_time_score=80 + i,
        )
        result = TestResult(
            model_name=f"model-{i}",
            question_number=1,
            question_text="Test question",
            context_provided=i % 2 == 0,
            response=f"Response {i}",
            response_time=10.0 + i,
            grades=grades,
        )
        results.append(result)

    return results


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("WATSONX_API_KEY", "test_watsonx_key")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test_project_id")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")


@pytest.fixture
def questions_dir(temp_dir):
    """Create a directory with sample question files."""
    questions_path = temp_dir / "questions"
    questions_path.mkdir()

    for i in range(1, 6):
        question_file = questions_path / f"question_{i}.txt"
        question_file.write_text(f"This is question {i}?")

    return questions_path


@pytest.fixture
def contexts_dir(temp_dir):
    """Create a directory with sample context files."""
    contexts_path = temp_dir / "contexts"
    contexts_path.mkdir()

    for i in range(1, 6):
        context_file = contexts_path / f"context_{i}.txt"
        context_file.write_text(f"This is context for question {i}.")

    return contexts_path
