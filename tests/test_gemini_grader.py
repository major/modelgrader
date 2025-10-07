"""Tests for gemini grader module."""

import pytest

from modelgrader.gemini_grader import (
    _create_grading_prompt,
    _extract_explanation,
    _parse_grades,
)
from modelgrader.models import GradeBreakdown


class TestCreateGradingPrompt:
    """Tests for _create_grading_prompt function."""

    def test_create_grading_prompt_basic(self):
        """Test creating grading prompt with question and response."""
        question = "How do I configure firewalld?"
        response = "Use firewall-cmd to configure firewalld..."

        prompt = _create_grading_prompt(question, response)

        assert question in prompt
        assert response in prompt
        assert "ACCURACY" in prompt
        assert "COMPLETENESS" in prompt
        assert "CLARITY" in prompt
        assert "RESPONSE_TIME" not in prompt
        assert "CONTEXT PROVIDED TO MODEL:" not in prompt


class TestParseGrades:
    """Tests for _parse_grades function."""

    def test_parse_grades_valid_output(self):
        """Test parsing valid Gemini output."""
        grade_text = """
ACCURACY: 85
COMPLETENESS: 75
CLARITY: 80

Justification: The response was accurate and clear...
"""
        grades = _parse_grades(grade_text)

        assert grades.accuracy == 85
        assert grades.completeness == 75
        assert grades.clarity == 80

    def test_parse_grades_lowercase(self):
        """Test parsing grades with lowercase keys."""
        grade_text = """
accuracy: 70
completeness: 65
clarity: 75
"""
        grades = _parse_grades(grade_text)

        assert grades.accuracy == 70
        assert grades.completeness == 65
        assert grades.clarity == 75

    def test_parse_grades_with_extra_text(self):
        """Test parsing grades with surrounding text."""
        grade_text = """
Here are my grades for this response:

ACCURACY: 90
COMPLETENESS: 85
CLARITY: 88

The response was excellent because...
"""
        grades = _parse_grades(grade_text)

        assert grades.accuracy == 90
        assert grades.completeness == 85

    def test_parse_grades_clamps_to_max(self):
        """Test that grades over 100 are clamped to 100."""
        grade_text = """
ACCURACY: 150
COMPLETENESS: 110
CLARITY: 105
"""
        grades = _parse_grades(grade_text)

        # All should be clamped to 100
        assert grades.accuracy == 100
        assert grades.completeness == 100
        assert grades.clarity == 100

    def test_parse_grades_clamps_to_min(self):
        """Test that negative grades are clamped to 0."""
        grade_text = """
ACCURACY: -5
COMPLETENESS: -10
CLARITY: -2
"""
        grades = _parse_grades(grade_text)

        # All should be clamped to 0
        assert grades.accuracy == 0
        assert grades.completeness == 0
        assert grades.clarity == 0

    def test_parse_grades_missing_field_raises_error(self):
        """Test that missing required field raises ValueError."""
        grade_text = """
ACCURACY: 85
COMPLETENESS: 75
"""
        # Missing CLARITY

        with pytest.raises(ValueError, match="Could not parse grades"):
            _parse_grades(grade_text)

    @pytest.mark.parametrize(
        "accuracy,completeness,clarity",
        [
            (100, 100, 100),
            (0, 0, 0),
            (85, 75, 80),
            (50, 60, 55),
        ],
    )
    def test_parse_grades_various_values(
        self, accuracy, completeness, clarity
    ):
        """Test parsing various grade combinations."""
        grade_text = f"""
ACCURACY: {accuracy}
COMPLETENESS: {completeness}
CLARITY: {clarity}
"""
        grades = _parse_grades(grade_text)

        assert grades.accuracy == accuracy
        assert grades.completeness == completeness
        assert grades.clarity == clarity

    def test_parse_grades_includes_explanation(self):
        """Test that explanation is extracted from grade text."""
        grade_text = """
ACCURACY: 85
COMPLETENESS: 75
CLARITY: 80

The response was accurate but lacked some detail. It could have included more examples.
"""
        grades = _parse_grades(grade_text)

        assert grades.explanation
        assert "accurate" in grades.explanation.lower()
        assert len(grades.explanation) <= 200


class TestExtractExplanation:
    """Tests for _extract_explanation function."""

    def test_extract_explanation_from_justification(self):
        """Test extracting explanation from justification text."""
        grade_text = """
ACCURACY: 85
COMPLETENESS: 75
CLARITY: 80

The response was accurate but lacked some detail. It could have included more examples to improve completeness.
"""
        explanation = _extract_explanation(grade_text)

        assert "accurate" in explanation.lower()
        assert len(explanation) <= 200

    def test_extract_explanation_truncates_long_text(self):
        """Test that long explanations are truncated."""
        # Create a single very long sentence (no sentence breaks)
        long_text = "A" * 300
        grade_text = f"""
ACCURACY: 85
COMPLETENESS: 75
CLARITY: 80

{long_text}
"""
        explanation = _extract_explanation(grade_text)

        assert len(explanation) <= 200
        assert explanation.endswith("...")

    def test_extract_explanation_handles_missing_justification(self):
        """Test handling when no justification is present."""
        grade_text = """
ACCURACY: 85
COMPLETENESS: 75
CLARITY: 80
"""
        explanation = _extract_explanation(grade_text)

        assert explanation == "No explanation provided."


# Note: grade_response and configure_gemini require actual API calls to Google Gemini,
# so they would need mocking or integration tests. For unit tests, we test the
# helper functions that don't require API calls.
