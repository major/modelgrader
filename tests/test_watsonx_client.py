"""Tests for watsonx client module."""

import pytest

from modelgrader.watsonx_client import create_prompt


class TestCreatePrompt:
    """Tests for create_prompt function."""

    def test_create_prompt_without_context(self):
        """Test creating prompt without context."""
        question = "How do I configure firewalld?"
        prompt = create_prompt(question)

        assert "Red Hat Enterprise Linux" in prompt
        assert question in prompt
        assert "Context information:" not in prompt

    def test_create_prompt_with_context(self):
        """Test creating prompt with context."""
        question = "How do I configure firewalld?"
        context = "Firewalld is the default firewall management tool..."

        prompt = create_prompt(question, context)

        assert "Red Hat Enterprise Linux" in prompt
        assert question in prompt
        assert "Context information:" in prompt
        assert context in prompt

    @pytest.mark.parametrize(
        "question,context",
        [
            ("Question 1", None),
            ("Question 2", "Context 2"),
            ("How to use dnf?", "DNF is the package manager..."),
            ("Configure sudo", "Sudo allows users to run commands..."),
        ],
    )
    def test_create_prompt_various_inputs(self, question, context):
        """Test prompt creation with various inputs."""
        prompt = create_prompt(question, context)

        # Should always contain the question
        assert question in prompt

        # Should contain context only if provided
        if context:
            assert context in prompt
        else:
            assert "Context information:" not in prompt


# Note: Other watsonx_client functions (create_watsonx_client, list_available_models,
# query_model) require actual API credentials and connections, so they would need
# mocking or integration tests. For unit tests, we test the functions that don't
# require API calls.
