"""Tests for configuration module."""

import pytest

from modelgrader.config import Settings, parse_question_numbers


class TestSettings:
    """Tests for Settings class."""

    def test_settings_with_env_vars(self, mock_env_vars):
        """Test that settings load from environment variables."""
        settings = Settings()  # type: ignore[call-arg]

        assert settings.watsonx_api_key == "test_watsonx_key"
        assert settings.watsonx_project_id == "test_project_id"
        assert settings.gemini_api_key == "test_gemini_key"

    def test_settings_defaults(self, mock_env_vars, monkeypatch, tmp_path):
        """Test that settings have correct default values."""
        # Point to non-existent .env file to test defaults
        fake_env = tmp_path / "fake.env"
        monkeypatch.setattr(
            "modelgrader.config.Settings.model_config",
            {"env_file": str(fake_env), "env_file_encoding": "utf-8", "extra": "ignore"},
        )

        settings = Settings()  # type: ignore[call-arg]

        assert settings.watsonx_url == "https://us-south.ml.cloud.ibm.com"
        assert settings.gemini_model == "gemini-2.5-flash-lite"
        assert settings.output_csv_path == "llm_grading_results.csv"
        assert settings.questions_dir == "data/questions"
        assert settings.contexts_dir == "data/contexts"
        assert settings.request_timeout == 120
        assert settings.question_numbers == "1"


class TestParseQuestionNumbers:
    """Tests for parse_question_numbers function."""

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("1", [1]),
            ("1,2,3", [1, 2, 3]),
            ("1,2,3,4,5", [1, 2, 3, 4, 5]),
            ("5,3,1", [5, 3, 1]),  # Order preserved
            ("1, 2, 3", [1, 2, 3]),  # Spaces handled
            (" 1 , 2 , 3 ", [1, 2, 3]),  # Extra spaces
        ],
    )
    def test_parse_question_numbers_valid(self, input_str, expected):
        """Test parsing valid question number strings."""
        result = parse_question_numbers(input_str)
        assert result == expected

    def test_parse_question_numbers_empty(self):
        """Test parsing empty string returns empty list."""
        result = parse_question_numbers("")
        assert result == []

    def test_parse_question_numbers_with_empty_elements(self):
        """Test that empty elements are filtered out."""
        result = parse_question_numbers("1,,2,  ,3")
        assert result == [1, 2, 3]
