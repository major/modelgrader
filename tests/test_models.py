"""Tests for data models."""

import pytest

from modelgrader.models import GradeBreakdown, Question, TestResult, calculate_percentiles


class TestQuestion:
    """Tests for Question model."""

    def test_question_creation(self, sample_question):
        """Test creating a question."""
        assert sample_question.number == 1
        assert "SELinux" in sample_question.text
        assert sample_question.context_path is not None

    def test_load_context(self, sample_question):
        """Test loading context from file."""
        context = sample_question.load_context()
        assert "SELinux troubleshooting" in context

    def test_load_context_missing_file(self, temp_dir):
        """Test loading context when file doesn't exist."""
        question = Question(
            number=1,
            text="Test question",
            context_path=temp_dir / "nonexistent.txt",
        )
        context = question.load_context()
        assert context == ""

    def test_load_context_no_path(self):
        """Test loading context when path is None."""
        question = Question(
            number=1,
            text="Test question",
            context_path=None,
        )
        context = question.load_context()
        assert context == ""


class TestGradeBreakdown:
    """Tests for GradeBreakdown model."""

    @pytest.mark.parametrize(
        "accuracy,completeness,clarity,expected_total",
        [
            (100, 100, 100, 100.0),  # Perfect score
            (50, 50, 50, 50.0),  # All 50%
            (80, 70, 60, 72.5),  # Mixed scores: 80*0.5 + 70*0.25 + 60*0.25 = 72.5
            (0, 0, 0, 0.0),  # Zero score
        ],
    )
    def test_weighted_score_calculation(
        self, accuracy, completeness, clarity, expected_total
    ):
        """Test weighted score calculation with various inputs."""
        grades = GradeBreakdown(
            accuracy=accuracy,
            completeness=completeness,
            clarity=clarity,
        )
        # Weighted: 50% accuracy + 25% completeness + 25% clarity
        expected = accuracy * 0.5 + completeness * 0.25 + clarity * 0.25
        assert grades.weighted_score == round(expected, 2)
        assert grades.total == grades.weighted_score

    def test_grade_validation_ranges(self):
        """Test that grade values are validated."""
        with pytest.raises(ValueError):
            GradeBreakdown(
                accuracy=101,  # Over 100
                completeness=50,
                clarity=50,
            )

        with pytest.raises(ValueError):
            GradeBreakdown(
                accuracy=50,
                completeness=-1,  # Negative
                clarity=50,
            )


class TestTestResult:
    """Tests for TestResult model."""

    def test_test_result_creation(self, sample_test_result):
        """Test creating a test result."""
        assert sample_test_result.model_name == "ibm/granite-3.1-8b-instruct"
        assert sample_test_result.question_number == 1
        assert sample_test_result.context_provided is True
        assert sample_test_result.response_time == 5.2

    def test_total_score_property(self, sample_test_result):
        """Test total_score property returns weighted score."""
        assert sample_test_result.total_score == sample_test_result.grades.total

    def test_to_csv_row(self, sample_test_result):
        """Test converting to CSV row format."""
        row = sample_test_result.to_csv_row()

        assert row["Model Name"] == "ibm/granite-3.1-8b-instruct"
        assert row["Question"] == "Q1"
        assert row["Context Provided"] == "Yes"
        assert row["Accuracy Score"] == 85
        assert row["Completeness Score"] == 75
        assert row["Clarity Score"] == 80
        assert row["Response time"] == 5.2
        assert "Weighted Score" in row
        assert "Percentile Rank" in row

    def test_to_csv_row_no_context(self, sample_grade_breakdown):
        """Test CSV row with no context."""
        result = TestResult(
            model_name="test-model",
            question_number=2,
            question_text="Test",
            context_provided=False,
            response="Response",
            response_time=3.0,
            grades=sample_grade_breakdown,
        )
        row = result.to_csv_row()
        assert row["Context Provided"] == "No"


class TestCalculatePercentiles:
    """Tests for calculate_percentiles function."""

    def test_calculate_percentiles_single_result(self, sample_test_result):
        """Test percentile calculation with single result."""
        results = [sample_test_result]
        results_with_percentiles = calculate_percentiles(results)

        assert len(results_with_percentiles) == 1
        # Single result should have percentile of 50
        assert results_with_percentiles[0].percentile == 50.0

    def test_calculate_percentiles_multiple_results(self, multiple_test_results):
        """Test percentile calculation with multiple results."""
        results = calculate_percentiles(multiple_test_results)

        # Check that percentiles are calculated
        assert all(r.percentile >= 0 for r in results)
        assert all(r.percentile <= 100 for r in results)

        # Lowest score should have lowest percentile
        sorted_by_score = sorted(results, key=lambda r: r.total_score)
        sorted_by_percentile = sorted(results, key=lambda r: r.percentile)

        # First and last should match (lowest score = lowest percentile)
        assert sorted_by_score[0].percentile == sorted_by_percentile[0].percentile
        assert sorted_by_score[-1].percentile == sorted_by_percentile[-1].percentile

    def test_calculate_percentiles_empty_list(self):
        """Test percentile calculation with empty list."""
        results = calculate_percentiles([])
        assert results == []

    def test_calculate_percentiles_distribution(self, multiple_test_results):
        """Test that percentiles are properly distributed."""
        results = calculate_percentiles(multiple_test_results)

        percentiles = [r.percentile for r in results]

        # Should have some spread in percentiles
        assert min(percentiles) == 0.0
        assert max(percentiles) == 100.0
        # Should have values in between
        middle_percentiles = [p for p in percentiles if 0 < p < 100]
        assert len(middle_percentiles) > 0
