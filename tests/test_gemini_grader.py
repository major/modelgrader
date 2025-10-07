"""Tests for gemini grader module."""

import pytest

from modelgrader.gemini_grader import _create_grading_prompt, _parse_grades
from modelgrader.models import GradeBreakdown


class TestCreateGradingPrompt:
    """Tests for _create_grading_prompt function."""

    def test_create_grading_prompt_without_context(self):
        """Test creating grading prompt without context."""
        question = "How do I configure firewalld?"
        response = "Use firewall-cmd to configure firewalld..."
        response_time = 5.5

        prompt = _create_grading_prompt(question, response, None, response_time)

        assert question in prompt
        assert response in prompt
        assert f"{response_time:.2f}" in prompt
        assert "CONTEXT PROVIDED TO MODEL:" not in prompt
        assert "ACCURACY" in prompt
        assert "COMPLETENESS" in prompt
        assert "CLARITY" in prompt
        assert "RESPONSE_TIME" in prompt

    def test_create_grading_prompt_with_context(self):
        """Test creating grading prompt with context."""
        question = "How do I configure firewalld?"
        response = "Use firewall-cmd..."
        context = "Firewalld is the default firewall..."
        response_time = 3.2

        prompt = _create_grading_prompt(question, response, context, response_time)

        assert question in prompt
        assert response in prompt
        assert context in prompt
        assert f"{response_time:.2f}" in prompt
        assert "CONTEXT PROVIDED TO MODEL:" in prompt

    @pytest.mark.parametrize(
        "response_time",
        [5.0, 15.0, 35.0, 75.0, 105.0, 150.0],
    )
    def test_create_grading_prompt_various_times(self, response_time):
        """Test prompt includes response time correctly."""
        prompt = _create_grading_prompt(
            "Question", "Response", None, response_time
        )

        assert f"{response_time:.2f}" in prompt


class TestParseGrades:
    """Tests for _parse_grades function."""

    def test_parse_grades_valid_output(self):
        """Test parsing valid Gemini output."""
        grade_text = """
ACCURACY: 85
COMPLETENESS: 75
CLARITY: 80
RESPONSE_TIME: 90

Justification: The response was accurate and clear...
"""
        grades = _parse_grades(grade_text)

        assert grades.accuracy == 85
        assert grades.completeness == 75
        assert grades.clarity == 80
        assert grades.response_time_score == 90

    def test_parse_grades_lowercase(self):
        """Test parsing grades with lowercase keys."""
        grade_text = """
accuracy: 70
completeness: 65
clarity: 75
response_time: 85
"""
        grades = _parse_grades(grade_text)

        assert grades.accuracy == 70
        assert grades.completeness == 65
        assert grades.clarity == 75
        assert grades.response_time_score == 85

    def test_parse_grades_with_extra_text(self):
        """Test parsing grades with surrounding text."""
        grade_text = """
Here are my grades for this response:

ACCURACY: 90
COMPLETENESS: 85
CLARITY: 88
RESPONSE_TIME: 95

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
RESPONSE_TIME: 120
"""
        grades = _parse_grades(grade_text)

        # All should be clamped to 100
        assert grades.accuracy == 100
        assert grades.completeness == 100
        assert grades.clarity == 100
        assert grades.response_time_score == 100

    def test_parse_grades_clamps_to_min(self):
        """Test that negative grades are clamped to 0."""
        grade_text = """
ACCURACY: -5
COMPLETENESS: -10
CLARITY: -2
RESPONSE_TIME: -8
"""
        grades = _parse_grades(grade_text)

        # All should be clamped to 0
        assert grades.accuracy == 0
        assert grades.completeness == 0
        assert grades.clarity == 0
        assert grades.response_time_score == 0

    def test_parse_grades_missing_field_raises_error(self):
        """Test that missing required field raises ValueError."""
        grade_text = """
ACCURACY: 85
COMPLETENESS: 75
CLARITY: 80
"""
        # Missing RESPONSE_TIME

        with pytest.raises(ValueError, match="Could not parse grades"):
            _parse_grades(grade_text)

    @pytest.mark.parametrize(
        "accuracy,completeness,clarity,response_time",
        [
            (100, 100, 100, 100),
            (0, 0, 0, 0),
            (85, 75, 80, 90),
            (50, 60, 55, 70),
        ],
    )
    def test_parse_grades_various_values(
        self, accuracy, completeness, clarity, response_time
    ):
        """Test parsing various grade combinations."""
        grade_text = f"""
ACCURACY: {accuracy}
COMPLETENESS: {completeness}
CLARITY: {clarity}
RESPONSE_TIME: {response_time}
"""
        grades = _parse_grades(grade_text)

        assert grades.accuracy == accuracy
        assert grades.completeness == completeness
        assert grades.clarity == clarity
        assert grades.response_time_score == response_time


# Note: grade_response and configure_gemini require actual API calls to Google Gemini,
# so they would need mocking or integration tests. For unit tests, we test the
# helper functions that don't require API calls.
