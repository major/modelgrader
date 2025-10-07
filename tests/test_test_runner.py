"""Tests for test runner module."""

import pytest

from modelgrader.test_runner import load_questions


class TestLoadQuestions:
    """Tests for load_questions function."""

    def test_load_questions(self, questions_dir, contexts_dir):
        """Test loading questions with contexts."""
        questions = load_questions(questions_dir, contexts_dir)

        assert len(questions) == 5
        # Check questions are in order
        for i, question in enumerate(questions, start=1):
            assert question.number == i
            assert f"question {i}" in question.text
            assert question.context_path is not None
            assert question.context_path.exists()

    def test_load_questions_context_content(self, questions_dir, contexts_dir):
        """Test that context can be loaded."""
        questions = load_questions(questions_dir, contexts_dir)

        for question in questions:
            context = question.load_context()
            assert len(context) > 0
            assert f"context for question {question.number}" in context

    def test_load_questions_missing_questions_dir(self, temp_dir):
        """Test error when questions directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Questions directory not found"):
            load_questions(temp_dir / "nonexistent", temp_dir)

    def test_load_questions_missing_contexts_dir(self, questions_dir, temp_dir):
        """Test error when contexts directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Contexts directory not found"):
            load_questions(questions_dir, temp_dir / "nonexistent")

    def test_load_questions_missing_context_file(self, questions_dir, temp_dir):
        """Test that missing context file is handled gracefully."""
        # Create contexts dir but no context files
        contexts_dir = temp_dir / "contexts"
        contexts_dir.mkdir()

        questions = load_questions(questions_dir, contexts_dir)

        # Questions should still be loaded, but context paths may not exist
        assert len(questions) == 5
        for question in questions:
            # Context path is set but file doesn't exist
            if question.context_path:
                # load_context should return empty string
                assert question.load_context() == ""

    def test_load_questions_sorted_by_number(self, temp_dir):
        """Test that questions are sorted by number."""
        questions_path = temp_dir / "questions"
        contexts_path = temp_dir / "contexts"
        questions_path.mkdir()
        contexts_path.mkdir()

        # Create questions out of order
        for num in [3, 1, 5, 2, 4]:
            (questions_path / f"question_{num}.txt").write_text(f"Question {num}")
            (contexts_path / f"context_{num}.txt").write_text(f"Context {num}")

        questions = load_questions(questions_path, contexts_path)

        # Should be sorted by number
        assert [q.number for q in questions] == [1, 2, 3, 4, 5]

    @pytest.mark.parametrize("num_questions", [1, 3, 10])
    def test_load_questions_various_counts(self, temp_dir, num_questions):
        """Test loading various numbers of questions."""
        questions_path = temp_dir / "questions"
        contexts_path = temp_dir / "contexts"
        questions_path.mkdir()
        contexts_path.mkdir()

        for i in range(1, num_questions + 1):
            (questions_path / f"question_{i}.txt").write_text(f"Question {i}")
            (contexts_path / f"context_{i}.txt").write_text(f"Context {i}")

        questions = load_questions(questions_path, contexts_path)

        assert len(questions) == num_questions


# Note: run_single_test and run_all_tests require actual API clients, so they would
# need mocking or integration tests. For unit tests, we test load_questions which
# doesn't require API calls.
