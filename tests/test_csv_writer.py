"""Tests for CSV writer module."""

import csv

import pytest

from modelgrader.csv_writer import (
    CSV_FIELDNAMES,
    append_result_to_csv,
    initialize_csv,
    load_all_results,
    load_existing_results,
    write_results_to_csv,
)


class TestInitializeCSV:
    """Tests for initialize_csv function."""

    def test_initialize_creates_file(self, temp_csv_file):
        """Test that initialize creates a file with headers."""
        # File doesn't exist yet (temp_csv_file is just a path)
        initialize_csv(temp_csv_file)

        assert temp_csv_file.exists()

        # Check that headers are written
        with temp_csv_file.open("r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert headers == CSV_FIELDNAMES

    def test_initialize_doesnt_overwrite_existing(self, temp_csv_file):
        """Test that initialize doesn't overwrite existing file."""
        # Write some content
        temp_csv_file.write_text("existing content")

        initialize_csv(temp_csv_file)

        # Content should be unchanged
        assert temp_csv_file.read_text() == "existing content"


class TestAppendResultToCSV:
    """Tests for append_result_to_csv function."""

    def test_append_result(self, temp_csv_file, sample_test_result):
        """Test appending a result to CSV."""
        initialize_csv(temp_csv_file)

        append_result_to_csv(sample_test_result, temp_csv_file)

        # Check file has one data row
        with temp_csv_file.open("r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["Model Name"] == "ibm/granite-3.1-8b-instruct"

    def test_append_multiple_results(self, temp_csv_file, multiple_test_results):
        """Test appending multiple results."""
        initialize_csv(temp_csv_file)

        for result in multiple_test_results[:3]:
            append_result_to_csv(result, temp_csv_file)

        # Check file has three data rows
        with temp_csv_file.open("r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3


class TestLoadExistingResults:
    """Tests for load_existing_results function."""

    def test_load_existing_results_empty_file(self, temp_csv_file):
        """Test loading from non-existent file."""
        # File doesn't exist yet (temp_csv_file is just a path)
        results = load_existing_results(temp_csv_file)

        assert results == set()

    def test_load_existing_results(self, temp_csv_file, multiple_test_results):
        """Test loading existing results."""
        initialize_csv(temp_csv_file)

        # Append some results
        for result in multiple_test_results[:3]:
            append_result_to_csv(result, temp_csv_file)

        # Load them back
        existing = load_existing_results(temp_csv_file)

        # Should have 3 tuples (model_name, question_number, context_provided)
        assert len(existing) == 3
        assert ("model-0", 1, True) in existing
        assert ("model-1", 1, False) in existing
        assert ("model-2", 1, True) in existing

    def test_load_existing_results_unique_combinations(
        self, temp_csv_file, sample_test_result
    ):
        """Test that duplicate combinations are deduplicated."""
        initialize_csv(temp_csv_file)

        # Append same result twice
        append_result_to_csv(sample_test_result, temp_csv_file)
        append_result_to_csv(sample_test_result, temp_csv_file)

        existing = load_existing_results(temp_csv_file)

        # Should only have one unique combination
        assert len(existing) == 1


class TestLoadAllResults:
    """Tests for load_all_results function."""

    def test_load_all_results_empty_file(self, temp_csv_file):
        """Test loading from non-existent file."""
        # File doesn't exist yet (temp_csv_file is just a path)
        results = load_all_results(temp_csv_file)

        assert results == []

    def test_load_all_results(self, temp_csv_file, multiple_test_results):
        """Test loading all results from CSV."""
        initialize_csv(temp_csv_file)

        # Append results
        for result in multiple_test_results[:3]:
            append_result_to_csv(result, temp_csv_file)

        # Load them back
        loaded_results = load_all_results(temp_csv_file)

        assert len(loaded_results) == 3
        # Check first result
        assert loaded_results[0].model_name == "model-0"
        assert loaded_results[0].question_number == 1
        assert loaded_results[0].context_provided is True

    def test_load_all_results_preserves_grades(
        self, temp_csv_file, sample_test_result
    ):
        """Test that grades are preserved when loading."""
        initialize_csv(temp_csv_file)
        append_result_to_csv(sample_test_result, temp_csv_file)

        loaded_results = load_all_results(temp_csv_file)

        assert len(loaded_results) == 1
        result = loaded_results[0]
        assert result.grades.accuracy == 85
        assert result.grades.completeness == 75
        assert result.grades.clarity == 80


class TestWriteResultsToCSV:
    """Tests for write_results_to_csv function."""

    def test_write_results_empty_list(self, temp_csv_file):
        """Test writing empty list doesn't create file."""
        # File doesn't exist yet (temp_csv_file is just a path)
        write_results_to_csv([], temp_csv_file)

        # File should not be created for empty list
        assert not temp_csv_file.exists()

    def test_write_results_overwrites_file(self, temp_csv_file, multiple_test_results):
        """Test that write_results_to_csv overwrites existing file."""
        # Write some initial content
        temp_csv_file.write_text("old content")

        write_results_to_csv(multiple_test_results[:3], temp_csv_file)

        # Check file is overwritten with new content
        with temp_csv_file.open("r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3

    def test_write_results_all_fields(self, temp_csv_file, sample_test_result):
        """Test that all fields are written correctly."""
        write_results_to_csv([sample_test_result], temp_csv_file)

        with temp_csv_file.open("r") as f:
            reader = csv.DictReader(f)
            row = next(reader)

            # Check all expected fields are present
            for field in CSV_FIELDNAMES:
                assert field in row

            assert row["Model Name"] == "ibm/granite-3.1-8b-instruct"
            assert row["Question"] == "Q1"
            assert row["Context Provided"] == "Yes"
            assert float(row["Percentile Rank"]) == 75.5

    @pytest.mark.parametrize("num_results", [1, 5, 10])
    def test_write_results_various_sizes(
        self, temp_csv_file, multiple_test_results, num_results
    ):
        """Test writing various numbers of results."""
        write_results_to_csv(multiple_test_results[:num_results], temp_csv_file)

        with temp_csv_file.open("r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == num_results
