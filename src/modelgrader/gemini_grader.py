"""Google Gemini-based grading system for LLM responses."""

import json
from typing import TypedDict

import google.generativeai as genai

from modelgrader.logging import get_logger
from modelgrader.models import GradeBreakdown

logger = get_logger(__name__)


class GradingResponse(TypedDict):
    """Structured response schema for Gemini grading."""

    accuracy: int
    completeness: int
    clarity: int
    explanation: str


def configure_gemini(api_key: str) -> None:
    """Configure the Gemini API.

    Args:
        api_key: Google Gemini API key
    """
    genai.configure(api_key=api_key)  # type: ignore[attr-defined]
    logger.info("gemini_configured")


def grade_response(
    question: str,
    response: str,
    context: str | None,
    response_time: float,
    model_name: str = "gemini-2.0-flash-exp",
) -> GradeBreakdown:
    """Grade an LLM response using Gemini.

    Args:
        question: The original question
        response: The model's response to grade
        context: Optional context that was provided (not used in grading)
        response_time: Time taken to generate response (not used in grading)
        model_name: Gemini model to use for grading

    Returns:
        GradeBreakdown with scores
    """
    logger.info(
        "grading_response",
        question_length=len(question),
        response_length=len(response),
        has_context=context is not None,
        response_time=round(response_time, 2),
    )

    # Create grading prompt (context not included in grading)
    grading_prompt = _create_grading_prompt(question, response)

    try:
        # Use Gemini to grade the response with structured output
        model = genai.GenerativeModel(  # type: ignore[attr-defined]
            model_name,
            generation_config={  # type: ignore[arg-type]
                "response_mime_type": "application/json",
                "response_schema": GradingResponse,
            },
        )
        result = model.generate_content(grading_prompt)  # type: ignore[attr-defined]

        # Parse JSON response directly
        grading_data = json.loads(result.text)  # type: ignore[attr-defined]

        logger.debug("parsed_grading_data", data=grading_data)

        # Validate required fields are present
        required_fields = ["accuracy", "completeness", "clarity"]
        missing_fields = [f for f in required_fields if f not in grading_data]
        if missing_fields:
            logger.warning(
                "missing_required_fields",
                fields=missing_fields,
                response=result.text[:500],  # type: ignore[attr-defined]
                available_keys=list(grading_data.keys()),
            )
            # Provide default values for missing fields
            for field in missing_fields:
                grading_data[field] = 50  # Default to middle score

        # Validate and clamp scores
        accuracy = min(100, max(0, int(grading_data.get("accuracy", 50))))
        completeness = min(100, max(0, int(grading_data.get("completeness", 50))))
        clarity = min(100, max(0, int(grading_data.get("clarity", 50))))

        grades = GradeBreakdown(
            accuracy=accuracy,
            completeness=completeness,
            clarity=clarity,
            explanation=grading_data.get("explanation", "No explanation provided."),
        )

        logger.info(
            "grading_complete",
            accuracy=grades.accuracy,
            completeness=grades.completeness,
            clarity=grades.clarity,
            total=grades.total,
        )

        return grades

    except Exception as e:
        logger.error("grading_failed", error=str(e))
        raise


def _create_grading_prompt(question: str, response: str) -> str:
    """Create a detailed grading prompt for Gemini.

    Args:
        question: The original question
        response: The model's response

    Returns:
        Formatted grading prompt
    """
    return f"""You are a STRICT expert grader evaluating LLM responses about Red Hat Enterprise Linux system administration.

CRITICAL INSTRUCTIONS:
1. BE CRITICAL - Use the full range of scores to differentiate responses
2. BE CONSISTENT - Apply the same standards to every response you grade
3. CALIBRATE - Use the grading rubric precisely; same issues = same point deductions every time

ORIGINAL QUESTION:
{question}

MODEL'S RESPONSE:
{response}

Grade this response on a 100-point scale for each category. IMPORTANT: Scores above 85 should be RARE and only given to exceptional responses. Most good responses should fall in the 60-80 range.

STRICT Grading scale (apply consistently to all responses):
- Exceptional (rare): 85-100 - Perfect or near-perfect, comprehensive, zero issues
- Very good: 70-84 - Solid response with minor room for improvement
- Good/Adequate: 55-69 - Correct but lacking in some way (detail, examples, completeness)
- Below average: 40-54 - Partially correct with notable gaps or minor errors
- Poor: 20-39 - Significant errors or missing critical information
- Severely deficient: 0-19 - Fundamentally wrong or unhelpful

Grade each category independently using CONSISTENT criteria:

1. ACCURACY (0-100): Does the response correctly address the user's question?
   - ANY technical errors should result in score below 70
   - Missing important caveats or warnings: deduct 10-15 points
   - Outdated or non-RHEL-specific info: deduct 10-20 points
   - Perfect accuracy with all edge cases covered: 85-100
   - Correct but missing some nuance: 60-75
   - Minor technical errors: 40-60
   - Major technical errors: 0-40

2. COMPLETENESS (0-100): Does the response provide a thorough answer?
   - Score above 80 ONLY if ALL aspects covered thoroughly with examples
   - Missing any significant aspect of the question: max 65
   - Missing examples when they would be helpful: deduct 10 points
   - Lacks context or explanation: deduct 10-15 points
   - Partial answer only: max 50
   - Superficial coverage: 30-50

3. CLARITY (0-100): Is the response well-written and easy to understand?
   - Score above 80 ONLY for exceptional organization and presentation
   - Poor structure or hard to follow: max 60
   - Lacks organization (no bullets, paragraphs, etc.): deduct 15 points
   - Verbose or unclear language: deduct 10 points
   - Missing helpful formatting: deduct 5-10 points
   - Professional but not exceptional: 60-75

CRITICAL GRADING RULES FOR CONSISTENCY:
- Start by assuming a baseline of 70, then deduct points for any issues
- Only exceptional responses with zero flaws should score above 85
- If you're unsure between two scores, choose the LOWER one
- Look for reasons to deduct points, not reasons to add them
- Apply the EXACT SAME deductions for similar issues across all responses
- Do not let earlier scores influence current grading - judge each response independently against the rubric
- Be mechanical and systematic: same flaw = same point deduction, every time

CONSISTENCY CHECKLIST:
Before assigning scores, ask yourself:
1. Am I applying the same severity of judgment as I would to any other response?
2. Would I deduct the same points if I saw this exact issue in a different response?
3. Am I being influenced by the overall quality, or grading each dimension independently?

Provide your response as a JSON object with these fields:
- accuracy: integer score 0-100
- completeness: integer score 0-100
- clarity: integer score 0-100
- explanation: brief justification (1-2 sentences) for the scores, explicitly noting what prevented higher scores and what specific deductions were applied

Note: These scores will be weighted as follows for the final grade:
- Accuracy: 50%
- Completeness: 25%
- Clarity: 25%"""


