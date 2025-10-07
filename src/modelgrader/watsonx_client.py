"""IBM WatsonX client for listing and querying models."""

import time

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models.inference import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from modelgrader.logging import get_logger

logger = get_logger(__name__)


def create_watsonx_client(api_key: str, project_id: str, url: str) -> APIClient:
    """Create and return a WatsonX API client.

    Args:
        api_key: IBM WatsonX API key
        project_id: WatsonX project ID
        url: WatsonX API URL

    Returns:
        Configured APIClient instance
    """
    credentials = Credentials(api_key=api_key, url=url)  # type: ignore[call-arg]
    client = APIClient(credentials=credentials, project_id=project_id)  # type: ignore[call-arg]
    logger.info("watsonx_client_created", url=url)
    return client


def list_available_models(client: APIClient) -> list[str]:
    """List chat-capable models in WatsonX, excluding non-chat and visual models.

    Args:
        client: WatsonX API client

    Returns:
        Sorted list of chat-capable model IDs
    """
    logger.info("listing_available_models")

    # Get all available foundation models
    models_info = client.foundation_models.get_model_specs()  # type: ignore[attr-defined]

    # Extract model IDs, filtering out non-text models
    model_ids = []
    visual_count = 0
    excluded_count = 0
    no_chat_count = 0
    deprecated_count = 0

    for model in models_info["resources"]:  # type: ignore[index]
        model_id = model["model_id"]

        # Check if model is deprecated
        lifecycle = model.get("lifecycle", [])
        lifecycle_ids = [item.get("id", "") for item in lifecycle]
        is_deprecated = "deprecated" in lifecycle_ids

        if is_deprecated:
            deprecated_count += 1
            logger.debug("skipping_deprecated_model", model_id=model_id)
            continue

        # Check if model supports chat completions
        tasks = model.get("tasks", [])
        task_ids = [task.get("id", "") for task in tasks]
        supports_chat = "chat" in task_ids or "question_answering" in task_ids

        # Also check model functions if available
        functions = model.get("functions", [])
        function_ids = [func.get("id", "") for func in functions]
        has_chat_function = "chat" in function_ids or "text_chat" in function_ids

        if not (supports_chat or has_chat_function):
            no_chat_count += 1
            logger.debug(
                "skipping_non_chat_model",
                model_id=model_id,
                tasks=task_ids,
                functions=function_ids,
            )
            continue

        # Check if model is visual/image-related
        # Visual models typically have these indicators in their ID or tasks
        model_id_lower = model_id.lower()
        is_visual = any([
            "vision" in model_id_lower,
            "visual" in model_id_lower,
            "image" in model_id_lower,
            "vlm" in model_id_lower,  # Vision-Language Model
            "clip" in model_id_lower,
            "vit" in model_id_lower,  # Vision Transformer
        ])

        # Also check the model's tasks field if available
        has_visual_task = any([
            "visual" in str(task).lower() or "image" in str(task).lower()
            for task in task_ids
        ])

        if is_visual or has_visual_task:
            visual_count += 1
            logger.debug("skipping_visual_model", model_id=model_id)
            continue

        # Exclude guardian models, llama-3-405b, and code-instruct models
        is_excluded = any([
            "guardian" in model_id_lower,
            "llama-3-405b" in model_id_lower,
            "code-instruct" in model_id_lower,
        ])

        if is_excluded:
            excluded_count += 1
            logger.debug("skipping_excluded_model", model_id=model_id)
            continue

        # Only text/chat generation models make it here
        model_ids.append(model_id)

    # Sort alphabetically
    model_ids.sort()

    logger.info(
        "models_listed",
        count=len(model_ids),
        deprecated_skipped=deprecated_count,
        no_chat_skipped=no_chat_count,
        visual_skipped=visual_count,
        excluded_skipped=excluded_count,
    )
    return model_ids


def query_model(
    client: APIClient,
    model_id: str,
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.7,
) -> tuple[str, float]:
    """Query a WatsonX model using chat completion and measure response time.

    Args:
        client: WatsonX API client
        model_id: Model ID to query
        prompt: Prompt to send to the model
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        Tuple of (response text, response time in seconds)
    """
    logger.info("querying_model", model_id=model_id, prompt_length=len(prompt))

    start_time = time.time()

    try:
        # Create model inference instance
        model = ModelInference(
            model_id=model_id,
            api_client=client,
            params={
                GenParams.MAX_NEW_TOKENS: max_tokens,
                GenParams.TEMPERATURE: temperature,
                GenParams.DECODING_METHOD: "greedy",
                GenParams.RANDOM_SEED: 42,
            },
        )

        # Use chat completion with messages format
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        # Generate response using chat
        chat_response = model.chat(messages=messages)

        # Extract text from chat response
        # The chat response returns a dict with 'choices' containing the message
        response_text = (
            chat_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )

        elapsed_time = time.time() - start_time

        logger.info(
            "model_query_success",
            model_id=model_id,
            response_time=round(elapsed_time, 2),
            response_length=len(response_text),
        )

        return response_text, elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(
            "model_query_failed",
            model_id=model_id,
            error=str(e),
            elapsed_time=round(elapsed_time, 2),
        )
        raise


def create_prompt(question: str, context: str | None = None) -> str:
    """Create a prompt for the model with optional context.

    Args:
        question: The question to ask
        context: Optional context information

    Returns:
        Formatted prompt
    """
    if context:
        return f"""
You are a helpful assistant that answers questions from system administrators using Red Hat Enterprise Linux.
Additional context information is provided below to help you answer the question.

Context information:
--------------------------------------------------
{context}
--------------------------------------------------


Question:
--------------------------------------------------
{question}
--------------------------------------------------

Please provide a clear, accurate, and complete answer based on the context provided and your existing training.
"""
    else:
        return f"""
You are a helpful assistant that answers questions from system administrators using Red Hat Enterprise Linux.

Question:
--------------------------------------------------
{question}
--------------------------------------------------

Please provide a clear, accurate, and complete answer.
"""
