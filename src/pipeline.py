from typing import Any
from openai import OpenAI

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "not-needed"  

# Names of the models used in LM Studio.
BASE_MODEL_NAME = "meta-llama-3-8b-instruct"
FINETUNED_MODEL_NAME = "theia-llama-3.1-8b-v1"
# Hardcoded list of default projects to look for in questions -> I don't use this anymore because its unfitting for dynamic app design
DEFAULT_PROJECTS = ["bitcoin", "ethereum", "ethereum_eip_150", "solana", "chainlink", "aave"]

# Type alias for a retrieved text chunk
Chunk = dict[str, Any]


def _get_lmstudio_client() -> "OpenAI":
    """Create (or return) a cached OpenAI client that talks to LM Studio"""

    global client
    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)
    return client


def call_llm_text(
    prompt: str, 
    model: str = BASE_MODEL_NAME, 
    temperature: float = 0.2
) -> str:
    """
    Call a chat-style LLM via LM Studio and return the assistant's text.

    Args:
        - prompt: full prompt text (instructions + question + context)
        - model: the model name as configured in LM Studio's server.
        - temperature: controls randomness (0.0 = deterministic, higher = more creative).

    Returns:
        - The model's reply as plain text

    Notes:
        This function assumes LM Studio is running with the local server enabled.
        If something goes wrong, it raises a RuntimeError with a message.
    """

    client = _get_lmstudio_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
    except Exception as e:  
        raise RuntimeError(
            f"Error while calling LM Studio model '{model}': {e}"
        ) from e

    message = response.choices[0].message
    content = message.content or ""
    return content.strip()


def analyze_question(
    question: str,
    available_projects: list[str] = DEFAULT_PROJECTS,
) -> dict[str, Any]:
    """
    Analyze the user's question.

    Args:
        - question: the defined question
        - available_projects: list of project IDs to detect in the question text (defaults to DEFAULT_PROJECTS)

    Returns a dictionary with:
        - "projects": list of project IDs that seem relevant
        - "type": type of question: 'overview', 'tokenomics', 'risk', 'comparison', ...
        - "needs_image": bool, whether a diagram might be helpful
    """

    q_lower = question.lower()

    # Very simple project detection: check if project appears in the text
    projects = [project for project in available_projects if project.lower() in q_lower]

    # Basic question type classification
    if "compare" in q_lower or "difference" in q_lower:
        q_type = "comparison"
    elif "risk" in q_lower or "risks" in q_lower:
        q_type = "risk"
    elif "tokenomic" in q_lower or "supply" in q_lower or "emission" in q_lower:
        q_type = "tokenomics"
    else:
        q_type = "overview"

    # Simple rule for when an image might help
    needs_image_keywords = [
        "image", "picture", "diagram", "visual", "chart", 
        "graph", "flow", "architecture", "distribution",
        "supply"]
    needs_image = any(word in q_lower for word in needs_image_keywords)

    analysis = {
        "projects": projects,  # Could be [] if the question does not mention any specific project
        "type": q_type,
        "needs_image": needs_image,
    }
    return analysis


def _build_answer_prompt(
    question: str,
    retrieved_chunks: list[Chunk],
) -> str:
    """
    Helper to build a clear prompt for the LLM.

    Args:
        - question: the defined question
        - retrieved_chunks: list of retrieved chunk dicts used as context 

    Returns:
        - prompt: a string that contains instructions, the question, and the chunk context
    
    """

    # Build a context string from the chunks 
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "")
        # Limit each chunk to avoid too long prompts
        snippet = text[:800]
        context_parts.append(f"[Chunk {i}]\n{snippet}")
    context_text = "\n\n".join(context_parts) if context_parts else "NO CONTEXT AVAILABLE"

    prompt = f"""
        You are a helpful assistant that explains crypto projects to students.
        You must strictly base your answer ONLY on the context chunks provided below.
        If the context is not sufficient to answer the question, say that clearly.

        Task:
        - Read the user's question.
        - Read the context chunks.
        - Write a clear, structured explanation.
        - Mention which chunks you are using by referencing their numbers (e.g. "see Chunk 1").
        - Do NOT invent facts that are not supported by the context.
        - Add a short disclaimer at the end that this is not financial advice.

        User question: {question}
        Context chunks: {context_text}

        Now write the answer:
        """.strip()
    
    return prompt


def generate_answer(
    question: str,
    retrieved_chunks: list[Chunk],
    use_finetuned_model: bool = False
) -> dict[str, Any]:
    """
    Generate an answer based on the retrieved chunks using an LLM.

    Args:
        - question : the defined question
        - retrieved_chunks : list of chunk dicts from the retrieval system. Each chunk must contain at least a "text" field.
        - use_finetuned_model : 
            - if False (default), use the base model (DEFAULT_LMSTUDIO_MODEL)
            - ifTrue, use the fine-tuned model (FINE_TUNED_MODEL_NAME)

    Returns a dictionary with: 
        - "answer_text": the generated answer as plain text
        - "citations": list of citation dicts (one per retrieved chunk), containing any available metadata
    """

    if not retrieved_chunks:
        answer_text = (
            "I could not find any relevant information in the documents for this question:\n"
            f" {question}\n\n"
            "Please try rephrasing your question or asking about another project."
        )
        citations: list[Any] = []
        return {"answer_text": answer_text, "citations": citations}
    if use_finetuned_model:
        model_name = FINETUNED_MODEL_NAME
    else:
        model_name = BASE_MODEL_NAME

    prompt = _build_answer_prompt(question, retrieved_chunks)
    # Call the LLM via LM Studio
    answer_text = call_llm_text(prompt, model=model_name)

    # Building a simple "citations" list from whatever metadata is there
    citations = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        citations.append({
            "chunk_number": index,         
            "project": chunk.get("project"),
            "doc_id": chunk.get("doc_id"),
            "chunk_id": chunk.get("chunk_id"),
        })

    return {
        "answer_text": answer_text,
        "citations": citations,
    }


def review_answer(
    question: str,
    retrieved_chunks: list[Chunk],
    draft_answer: dict[str, Any],
    use_llm: bool = False,
    use_finetuned_model: bool = False
) -> dict[str, Any]:
    """
    Optional second step to refine the answer using another LLM call.

    Args: 
        - question : the defined question
        - retrieved_chunks : same chunks used for the initial answer
        - draft_answer : the answer returned by generate_answer(...)
        - use_llm : 
            - if False (default), do nothing and return draft_answer unchanged
            - if True, call the LLM again to refine the answer (e.g. improve clarity or enforce a certain style)
        - use_finetuned_model: same idea as in generate_answer(...):
            - if True, use the fine-tuned model for the review step.
            - if False, use the base model.

    Returns a dictionary with:
        - refined answer object in the same format as draft_answer
    """

    if not use_llm:
        return draft_answer
    
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "")
        snippet = text[:400]
        context_parts.append(f"[Chunk {i}]\n{snippet}")
    context_text = "\n\n".join(context_parts) if context_parts else "NO CONTEXT AVAILABLE"

    draft_text = draft_answer.get("answer_text", "")

    review_prompt = f"""
        You are reviewing an answer about crypto projects.

        User question: {question}

        Context chunks (the only allowed sources of truth): {context_text}

        Draft answer: {draft_text}

        Task:
            - Check if the draft answer is consistent with the context.
            - Improve clarity and structure where needed.
            - Keep or improve the disclaimer about not giving financial advice.
            - If the draft contains information that is NOT supported by the context,
              remove or clearly mark it as uncertain.

        Now write the improved answer:
        """.strip()

    # Choosing the model (base vs. fine-tuned) for the review step
    if use_finetuned_model:
        model_name = FINETUNED_MODEL_NAME
    else:
        model_name = BASE_MODEL_NAME

    improved_text = call_llm_text(review_prompt, model=model_name)

    # Returning the same structure, but with updated answer_text
    refined = dict(draft_answer)
    refined["answer_text"] = improved_text
    return refined


def build_image_request(
    question: str,
    analysis: dict[str, Any],
    answer: dict[str, Any],
) -> dict[str, Any]:
    """
    Decide whether an image should be generated and suggest a prompt.

    This does NOT generate the image itself.
    Args:
        - question: the defined question
        - analysis: output from analyze_question(...), used to decide if an image is helpful
        - answer: answer object returned by generate_answer(...), used to help formulating a prompt suggestion

    Returns a dictionary with:
      - "should_generate": bool
      - "prompt_suggestion": str or None
    """

    needs_image = analysis.get("needs_image", False)
    if not needs_image:
        return {
            "should_generate": False,
            "prompt_suggestion": None,
        }
    
    answer_text = answer.get("answer_text", "")

    prompt = (
        f"I had the following question: '{question}'\n\n"
        f"and I got this answer: '{answer_text}"
        "Create a clear and informative visualization, which helps to answer the question together with the already provided answer text."
        "The visualization should support the answer text, to clarify and answer the question."
        "Use simple shapes and labels to create the desired image."
    )

    return {
        "should_generate": True,
        "prompt_suggestion": prompt,
    }
