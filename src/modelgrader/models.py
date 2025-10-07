"""Data models for the LLM grading system."""

from pathlib import Path

from pydantic import BaseModel, Field


class Question(BaseModel):
    """A test question with optional context."""

    number: int = Field(..., description="Question number")
    text: str = Field(..., description="Question text")
    context_path: Path | None = Field(default=None, description="Path to context file")

    def load_context(self) -> str:
        """Load context from file if available.

        Returns:
            Context text or empty string if no context
        """
        if self.context_path and self.context_path.exists():
            return self.context_path.read_text(encoding="utf-8")
        return ""


class GradeBreakdown(BaseModel):
    """Breakdown of grades for a response (absolute scores 0-100)."""

    accuracy: int = Field(
        ...,
        ge=0,
        le=100,
        description="Accuracy score (0-100)",
    )
    completeness: int = Field(
        ...,
        ge=0,
        le=100,
        description="Completeness score (0-100)",
    )
    clarity: int = Field(
        ...,
        ge=0,
        le=100,
        description="Clarity score (0-100)",
    )
    explanation: str = Field(
        default="",
        description="Grading justification (1-2 sentences)",
    )

    @property
    def weighted_score(self) -> float:
        """Calculate weighted score based on category weights.

        Weights: Accuracy 50%, Completeness 25%, Clarity 25%

        Returns:
            Weighted score (0-100)
        """
        return round(
            (self.accuracy * 0.5)
            + (self.completeness * 0.25)
            + (self.clarity * 0.25),
            2,
        )

    # Maintain backward compatibility with 'total' property
    @property
    def total(self) -> float:
        """Alias for weighted_score for backward compatibility."""
        return self.weighted_score


class TestResult(BaseModel):
    """Result of testing a model with a question."""

    model_name: str = Field(..., description="Name of the model tested")
    question_number: int = Field(..., description="Question number")
    question_text: str = Field(..., description="Question text")
    context_provided: bool = Field(..., description="Whether context was provided")
    response: str = Field(..., description="Model's response")
    response_time: float = Field(..., ge=0, description="Response time in seconds")
    grades: GradeBreakdown = Field(..., description="Grade breakdown")
    percentile: float = Field(default=0.0, description="Percentile rank (0-100)")

    @property
    def total_score(self) -> float:
        """Get weighted score.

        Returns:
            Weighted score (0-100)
        """
        return self.grades.total

    def to_csv_row(self) -> dict[str, str | int | float]:
        """Convert to CSV row format.

        Returns:
            Dictionary suitable for CSV writing
        """
        return {
            "Model Name": self.model_name,
            "Question": f"Q{self.question_number}",
            "Context Provided": "Yes" if self.context_provided else "No",
            "Accuracy Score": self.grades.accuracy,
            "Completeness Score": self.grades.completeness,
            "Clarity Score": self.grades.clarity,
            "Response time": round(self.response_time, 2),
            "Weighted Score": self.total_score,
            "Percentile Rank": round(self.percentile, 1),
            "Explanation": self.grades.explanation,
        }


def calculate_percentiles(results: list[TestResult]) -> list[TestResult]:
    """Calculate percentile ranks for all results based on weighted scores.

    Args:
        results: List of test results

    Returns:
        List of test results with percentiles calculated
    """
    if not results:
        return results

    # Sort by weighted score
    sorted_results = sorted(results, key=lambda r: r.total_score)

    # Calculate percentile for each result
    n = len(sorted_results)
    for i, result in enumerate(sorted_results):
        # Percentile rank formula: (i / (n - 1)) * 100
        # Where i is the rank (0-indexed)
        percentile = (i / (n - 1)) * 100 if n > 1 else 50.0
        result.percentile = round(percentile, 1)

    return results
