"""
This module contains the LLM pipeline logic for the project.

The pipeline:
1. Analyzes the user's question (which projects? what type of question?).
2. Asks the RAG/retrieval system for relevant text chunks.
3. Generates an answer based on those chunks (LLM later, dummy now). #########################################
4. Optionally refines the answer (second LLM call later).
5. Suggests whether an image should be generated, and with what prompt.

Important:
- We do NOT care how the RAG/index is implemented internally.
- We only expect a function: retrieve(question: str, k: int = 5) -> list[dict]
  where each dict has at least a "text" field.
"""

from typing import List, Dict, Any, Callable, Optional
from openai import OpenAI

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "not-needed"  

# Name of the model you start in LM Studio.
BASE_MODEL_NAME = "meta-llama-3-8b-instruct"
FINETUNED_MODEL_NAME = "theia-llama-3.1-8b-v1"
DEFAULT_PROJECTS = ["bitcoin", "ethereum", "ethereum_eip_150", "solana", "chainlink", "aave"]

# Type alias for a retrieved text chunk
Chunk = Dict[str, Any]

# Step 0: LLM helper 

def _get_lmstudio_client() -> "OpenAI":
    """
    Create (or return) a cached OpenAI client that talks to LM Studio.

    LM Studio exposes an OpenAI-compatible API when you start the
    "Local Server" in the app. We simply point the OpenAI client
    to that base_url.
    """
    global client
    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)
    return client

def call_llm_text(prompt: str, model: str = BASE_MODEL_NAME, temperature: float = 0.2) -> str:
    """
    Call a chat-style LLM via LM Studio and return the assistant's text.

    Parameters
    ----------
    prompt : str
        Full prompt text (instructions + question + context) that we send
        as a single user message.
    model : str
        The model name as configured in LM Studio's server.
    temperature : float
        Controls randomness (0.0 = deterministic, higher = more creative).

    Returns
    -------
    str
        The model's reply as plain text.

    Notes
    -----
    - This function assumes LM Studio is running with the local server enabled.
    - If something goes wrong, it raises a RuntimeError with a message.
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


# Step 1: Question analysis

def analyze_question(
    question: str,
    available_projects: List[str] = DEFAULT_PROJECTS,
) -> Dict[str, Any]:
    """
    Analyze the user's question.

    In the future:
      - This could call an LLM or a fine-tuned classifier.
    For now:
      - We use simple string checks (heuristics) so that the pipeline
        can already run.

    Returns a dictionary with:
      - 'projects': list of project IDs that seem relevant
      - 'type': type of question: 'overview', 'tokenomics', 'risk', 'comparison', ...
      - 'needs_image': bool, whether a diagram might be helpful
    """
    q_lower = question.lower()

    # Very simple project detection: check if project appears in the text
    # Example: available_projects = ["bitcoin", "ethereum", "uniswap_v2", ...]
    projects = [p for p in available_projects if p.lower() in q_lower]

    # Very simple question type classification
    if "compare" in q_lower or "difference" in q_lower:
        q_type = "comparison"
    elif "risk" in q_lower or "risks" in q_lower:
        q_type = "risk"
    elif "tokenomic" in q_lower or "supply" in q_lower or "emission" in q_lower:
        q_type = "tokenomics"
    else:
        q_type = "overview"

    # Very simple rule for when an image might help
    needs_image_keywords = [
        "image", "picture", "diagram", "visual", "chart", 
        "graph", "flow", "architecture", "distribution",
        "supply"]
    needs_image = any(word in q_lower for word in needs_image_keywords)

    analysis = {
        "projects": projects,        # Could be [] if the question does not mention any specific project
        "type": q_type,
        "needs_image": needs_image,
    }
    return analysis


# Step 2: Answer generation (LLM later, dummy now)

def _build_answer_prompt(
    question: str,
    retrieved_chunks: List[Chunk],
) -> str:
    """
    Helper to build a clear prompt for the LLM.

    Steps:
      - Explain the task.
      - Provide the question.
      - Provide the retrieved context chunks.
      - Ask the model to answer based ONLY on this context and to be clear
        that this is not financial advice.
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
    retrieved_chunks: List[Chunk],
    use_finetuned_model: bool = False
) -> Dict[str, Any]:
    """
    Generate an answer based on the retrieved chunks using an LLM.

    Parameters
    ----------
    question : str
        The user's question.
    retrieved_chunks : list[Chunk]
        List of chunk dicts from the retrieval system. Each chunk must
        contain at least a "text" field.
    use_finetuned_model : bool
        If False (default):
            Use the base model (DEFAULT_LMSTUDIO_MODEL).
        If True:
            Use the fine-tuned model (FINE_TUNED_MODEL_NAME).
            This assumes you have trained such a model separately and
            configured its name above.

    Behavior
    --------
    If no chunks are available a "no info" message and empty citations are returned.
    Otherwise a prompt with the question + context is built, the chosen LLM is called via LM Studio, 
    and the model's answer and simple citations are returned.

    Returns
    -------
    dict
        {
          "answer_text": str,
          "citations": list[dict],
        }
    """
    if not retrieved_chunks:
        answer_text = (
            "I could not find any relevant information in the documents for this question:\n"
            f" {question}\n\n"
            "Please try rephrasing your question or asking about another project."
        )
        citations: List[Any] = []
        return {"answer_text": answer_text, "citations": citations}
    if use_finetuned_model:
        model_name = FINETUNED_MODEL_NAME
    else:
        model_name = BASE_MODEL_NAME

    prompt = _build_answer_prompt(question, retrieved_chunks)
    # Call the LLM via LM Studio
    answer_text = call_llm_text(prompt, model=model_name)

    # Build simple "citations" from whatever metadata we have
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


# Step 3: Optional review

def review_answer(
    question: str,
    retrieved_chunks: List[Chunk],
    draft_answer: Dict[str, Any],
    use_llm: bool = False,
    use_finetuned_model: bool = False
) -> Dict[str, Any]:
    """
    Optional second step to refine the answer using another LLM call.

    Parameters
    ----------
    question : str
        The user's question.
    retrieved_chunks : list[Chunk]
        Same chunks used for the initial answer.
    draft_answer : dict
        The answer returned by generate_answer(...).
    use_llm : bool
        If False (default):
            Do nothing and return draft_answer unchanged.
        If True:
            Call the LLM again to refine the answer (e.g., improve clarity
            or enforce a certain style).
    use_finetuned_model : bool
        Same idea as in generate_answer(...):
        If True: use the fine-tuned model for the review step.
        If False: use the base model.

    Returns
    -------
    dict
        Refined answer object in the same format as draft_answer.
    """
    if not use_llm:
        return draft_answer
    
    # If we want to use the LLM for review, build a prompt that includes:
    # - the original question,
    # - the retrieved chunks (again),
    # - the draft answer.
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

    # Choose model (base vs fine-tuned) for the review step
    if use_finetuned_model:
        model_name = FINETUNED_MODEL_NAME
    else:
        model_name = BASE_MODEL_NAME

    improved_text = call_llm_text(review_prompt, model=model_name)

    # Return the same structure, but with updated answer_text.
    refined = dict(draft_answer)
    refined["answer_text"] = improved_text
    return refined


# Step 4: Build an image generation request -> for Person 4 (Rakesh)

def build_image_request(
    question: str,
    analysis: Dict[str, Any],
    answer: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Decide whether an image should be generated and suggest a prompt.

    This does NOT generate the image itself.
    It only creates a small dict with:
      - 'should_generate': bool
      - 'prompt_suggestion': str or None

    Person 4 (Rakesh) (imaging.py + notebook) will:
      - read this information, and
      - call generate_tokenomics_image(prompt) if needed.
    """
    needs_image = analysis.get("needs_image", False)
    if not needs_image:
        return {
            "should_generate": False,
            "prompt_suggestion": None,
        }
    
    answer_text = answer.get("answer_text", "")

    # Very simple prompt suggestion:
    # Later, we can make this smarter and include more info from the answer.
    #projects = analysis.get("projects", [])
    #proj_part = ", ".join(projects) if projects else "the mentioned project(s)"

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
