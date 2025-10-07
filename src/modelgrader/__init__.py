"""LLM Grading System - Main entry point."""

from modelgrader.config import load_settings, parse_question_numbers
from modelgrader.console_output import (
    add_result_to_table,
    create_progress_bar,
    create_results_table,
    print_error,
    print_header,
    print_models_info,
    print_questions_info,
    print_resume_info,
    print_results_table,
    print_success,
    print_summary,
)
from modelgrader.csv_writer import (
    append_result_to_csv,
    initialize_csv,
    load_all_results,
    load_existing_results,
    write_results_to_csv,
)
from modelgrader.gemini_grader import configure_gemini
from modelgrader.logging import configure_logging, get_logger
from modelgrader.models import calculate_percentiles
from modelgrader.test_runner import load_questions, run_single_test
from modelgrader.watsonx_client import create_watsonx_client, list_available_models

logger = get_logger(__name__)


def main() -> None:
    """Main entry point for the LLM grading system."""
    # Configure logging
    configure_logging(log_level="INFO")

    # Print header
    print_header()

    try:
        # Load configuration
        logger.info("loading_configuration")
        settings = load_settings()

        # Configure Gemini
        configure_gemini(settings.gemini_api_key)

        # Create WatsonX client
        watsonx_client = create_watsonx_client(
            api_key=settings.watsonx_api_key,
            project_id=settings.watsonx_project_id,
            url=settings.watsonx_url,
        )

        # List available models
        model_ids = list_available_models(watsonx_client)
        print_models_info(len(model_ids), model_ids)

        # Load questions
        all_questions = load_questions(settings.questions_dir, settings.contexts_dir)

        # Filter questions based on configuration
        question_nums_to_test = parse_question_numbers(settings.question_numbers)
        questions = [q for q in all_questions if q.number in question_nums_to_test]

        if not questions:
            raise ValueError(
                f"No questions found matching numbers: {settings.question_numbers}"
            )

        logger.info(
            "filtered_questions",
            total=len(all_questions),
            selected=len(questions),
            numbers=question_nums_to_test,
        )
        print_questions_info(len(questions))

        # Calculate total tests
        total_tests = len(model_ids) * len(questions) * 2  # x2 for with/without context

        # Initialize CSV file and load existing results
        initialize_csv(settings.output_csv_path)
        existing_results = load_existing_results(settings.output_csv_path)
        print_resume_info(len(existing_results), total_tests)

        # Create progress bar and results table
        progress = create_progress_bar()
        results_table = create_results_table()
        all_results = []

        with progress:
            task = progress.add_task(
                "[cyan]Testing models...", total=total_tests
            )

            # Run all tests
            for model_id in model_ids:
                for question in questions:
                    # Test without context
                    test_key = (model_id, question.number, False)
                    if test_key in existing_results:
                        # Skip already tested
                        logger.debug(
                            "skipping_existing_test",
                            model_id=model_id,
                            question=question.number,
                            with_context=False,
                        )
                        progress.advance(task)
                    else:
                        try:
                            result = run_single_test(
                                client=watsonx_client,
                                model_id=model_id,
                                question=question,
                                with_context=False,
                            )
                            all_results.append(result)
                            add_result_to_table(results_table, result)
                            # Append immediately to CSV
                            append_result_to_csv(result, settings.output_csv_path)
                            progress.advance(task)
                        except Exception as e:
                            logger.error(
                                "test_failed",
                                model_id=model_id,
                                question=question.number,
                                with_context=False,
                                error=str(e),
                            )
                            progress.advance(task)

                    # Test with context
                    test_key = (model_id, question.number, True)
                    if test_key in existing_results:
                        # Skip already tested
                        logger.debug(
                            "skipping_existing_test",
                            model_id=model_id,
                            question=question.number,
                            with_context=True,
                        )
                        progress.advance(task)
                    else:
                        try:
                            result = run_single_test(
                                client=watsonx_client,
                                model_id=model_id,
                                question=question,
                                with_context=True,
                            )
                            all_results.append(result)
                            add_result_to_table(results_table, result)
                            # Append immediately to CSV
                            append_result_to_csv(result, settings.output_csv_path)
                            progress.advance(task)
                        except Exception as e:
                            logger.error(
                                "test_failed",
                                model_id=model_id,
                                question=question.number,
                                with_context=True,
                                error=str(e),
                            )
                            progress.advance(task)

        # Load all results (including existing ones) for percentile calculation
        logger.info("loading_all_results_for_percentile_calculation")
        all_results = load_all_results(settings.output_csv_path)

        # Calculate percentiles for all results
        logger.info("calculating_percentiles", result_count=len(all_results))
        all_results = calculate_percentiles(all_results)

        # Print results table
        print_results_table(results_table)

        # Write CSV
        write_results_to_csv(all_results, settings.output_csv_path)
        print_success(f"Results saved to {settings.output_csv_path}")

        # Print summary
        print_summary(all_results)

    except Exception as e:
        logger.error("main_failed", error=str(e), exc_info=True)
        print_error(str(e))
        raise


if __name__ == "__main__":
    main()
