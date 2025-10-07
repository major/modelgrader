"""Configuration management using pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # IBM WatsonX Configuration
    watsonx_api_key: str = Field(
        ...,
        description="IBM WatsonX API key",
    )
    watsonx_project_id: str = Field(
        ...,
        description="IBM WatsonX project ID",
    )
    watsonx_url: str = Field(
        default="https://us-south.ml.cloud.ibm.com",
        description="IBM WatsonX API URL (Dallas region)",
    )

    # Google Gemini Configuration
    gemini_api_key: str = Field(
        ...,
        description="Google Gemini API key",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash-lite",
        description="Gemini model to use for grading",
    )

    # Application Configuration
    output_csv_path: str = Field(
        default="llm_grading_results.csv",
        description="Path to output CSV file",
    )
    questions_dir: str = Field(
        default="data/questions",
        description="Directory containing question files",
    )
    contexts_dir: str = Field(
        default="data/contexts",
        description="Directory containing context files",
    )
    request_timeout: int = Field(
        default=120,
        description="Timeout for API requests in seconds",
    )
    question_numbers: str = Field(
        default="5",
        description="Comma-separated list of question numbers to test (e.g., '1' or '1,2,3')",
    )


def load_settings() -> Settings:
    """Load and return application settings."""
    return Settings()  # type: ignore[call-arg]


def parse_question_numbers(question_numbers_str: str) -> list[int]:
    """Parse comma-separated question numbers string into a list of integers.

    Args:
        question_numbers_str: Comma-separated string like "1,2,3" or "1"

    Returns:
        List of question numbers as integers
    """
    return [int(num.strip()) for num in question_numbers_str.split(",") if num.strip()]
