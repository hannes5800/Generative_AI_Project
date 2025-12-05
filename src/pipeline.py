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

# Type alias for a retrieved text chunk
Chunk = Dict[str, Any]

# Type alias for the retrieve function that Person 2 (Timon) will provide
RetrieveFn = Callable[[str, int], List[Chunk]]

# Step 0: (Optional) LLM helper 

def call_llm_text(prompt: str) -> str:
    """
    Placeholder for a simple text-only LLM call.

    Later we can implement this using:
    - a cloud API (e.g. OpenAI), or
    - a local model (e.g. LM Studio with an OpenAI-compatible endpoint).

    For now, we just raise an error so we don't accidentally call it
    without implementing it properly.
    """
    raise NotImplementedError("call_llm_text is not implemented yet.")


# Step 1: Question analysis

def analyze_question(
    question: str,
    available_projects: List[str],
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
    needs_image_keywords = ["image", "picture", "diagram", "visual", "chart", "graph", "flow", "architecture", "distribution"]
    needs_image = any(word in q_lower for word in needs_image_keywords)

    analysis = {
        "projects": projects,        # Could be [] if the question does not mention any specific project
        "type": q_type,
        "needs_image": needs_image,
    }
    return analysis


# Step 2: Retrieval wrapper

def retrieve_for_question(
    question: str,
    analysis: Dict[str, Any],
    retrieve_fn: RetrieveFn,
    k: int = 5,
) -> List[Chunk]:
    """
    Ask the retrieval system (RAG) for relevant text chunks.

    Parameters
    ----------
    question : str
        The user's question.
    analysis : dict
        Output of analyze_question(...) ? we might use this later to
        improve retrieval (e.g. project-specific queries).
        For now, we just pass the original question to retrieve_fn.
    retrieve_fn : callable
        Function provided by Person 2 (Timon), signature:
            retrieve(question: str, k: int = 5) -> list[dict]
        Each dict must contain at least a "text" field.
    k : int
        Maximum number of chunks to retrieve.

    Returns
    -------
    List[Chunk]
        A list of chunk dicts (can be empty).
    """
    # NOTE:
    # Currently, we do NOT use analysis["projects"] here because
    # retrieve_fn does not support project filters yet.
    # If in the future Person 2 (Timon) extends retrieve_fn, we can easily
    # update this function to pass additional arguments.
    chunks = retrieve_fn(question, k)
    return chunks


# Step 3: Answer generation (LLM later, dummy now)

def generate_answer(
    question: str,
    retrieved_chunks: List[Chunk],
) -> Dict[str, Any]:
    """
    Generate an answer based on the retrieved chunks.

    Later:
      - This should call an LLM with a prompt that includes:
          * the user's question
          * the retrieved chunks as context
      - The LLM should produce a structured answer and cite the chunks.

    For now:
      - We create a dummy answer that simply:
          * prints the question
          * shows a preview of the first few chunks
      - This allows us to test the whole pipeline without an LLM.

    Returns a dictionary with:
      - 'answer_text': the answer as a string (dummy or LLM-generated)
      - 'citations': a list of simple references to the chunks
    """
    if not retrieved_chunks:
        answer_text = (
            "Sorry, I couldn't find any relevant information for this question:\n"
            f"  {question}"
        )
        citations: List[Any] = []
    else:
        # Build a short preview of the first 2 chunks (max 300 characters each)
        previews = []
        for chunk in retrieved_chunks[:2]:
            text_preview = chunk.get("text", "")[:300]
            previews.append(text_preview)

        joined_preview = "\n\n---\n\n".join(previews)
        answer_text = (
            "DUMMY ANSWER (no real LLM used yet)\n\n"
            f"Question:\n{question}\n\n"
            "Here are some relevant snippets from the documents:\n\n"
            f"{joined_preview}"
        )

        # Build simple "citations" from whatever metadata we have
        citations = []
        for chunk in retrieved_chunks:
            citations.append({
                "project": chunk.get("project"),
                "doc_id": chunk.get("doc_id"),
                "chunk_id": chunk.get("chunk_id"),
            })

    return {
        "answer_text": answer_text,
        "citations": citations,
    }


# Step 4: Optional review / refinement

def review_answer(
    question: str,
    retrieved_chunks: List[Chunk],
    draft_answer: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Optional second step to refine the answer using another LLM call.

    For now:
      - We simply return the draft answer unchanged.

    Later:
      - This function can call an LLM again and ask:
          * "Is the answer grounded in these chunks?"
          * "Can you improve clarity and add a disclaimer?"
      - It should return a refined answer object in the same format.
    """
    # Placeholder: no review yet
    return draft_answer


# Step 5: Build an image generation request -> for Person 4 (Rakesh)

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

    # Very simple prompt suggestion:
    # Later, we can make this smarter and include more info from the answer.
    projects = analysis.get("projects", [])
    proj_part = ", ".join(projects) if projects else "the mentioned project(s)"

    prompt = (
        "Create a clear, informative image that helps explain the topic of this question:\n"
        f"'{question}'\n\n"
        f"The diagram should focus on the crypto project(s): {proj_part}.\n"
        "Always consider if the image is related to the entire question."
        "Use simple shapes and labels to create the desired image."
    )

    return {
        "should_generate": True,
        "prompt_suggestion": prompt,
    }


# Step 6: Main pipeline entrypoint

def pipeline_answer(
    question: str,
    available_projects: List[str],
    retrieve_fn: RetrieveFn,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Main function used by the notebook and by Person 4 (Rakesh).

    Steps:
      1. Analyze the question (which projects, type, needs_image).
      2. Retrieve up to k relevant chunks via retrieve_fn.
      3. Generate an answer based on those chunks.
      4. Optionally refine the answer (currently a no-op).
      5. Build an image generation request.

    Parameters
    ----------
    question : str
        The user's question.
    available_projects : list[str]
        List of project IDs we support (e.g. ["bitcoin", "ethereum", ...]).
        Used only for analysis (not retrieval) at the moment.
    retrieve_fn : callable
        Function provided by Person 2 (Timon):
            retrieve(question: str, k: int = 5) -> list[dict]
    k : int
        Maximum number of chunks to retrieve.

    Returns
    -------
    dict
        {
          "answer_text": str,
          "citations": [...], 
          "analysis": {...},
          "image_request": {
              "should_generate": bool,
              "prompt_suggestion": str or None,
          },
        }
    """
    # 1) Analyze the question (simple heuristics for now)
    analysis = analyze_question(question, available_projects)

    # 2) Retrieve relevant chunks from the RAG system
    retrieved_chunks = retrieve_for_question(
        question=question,
        analysis=analysis,
        retrieve_fn=retrieve_fn,
        k=k,
    )

    # 3) Generate a draft answer (dummy text for now, LLM later)
    draft_answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    # 4) Optionally refine the answer (placeholder)
    final_answer = review_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        draft_answer=draft_answer,
    )

    # 5) Build image generation request for Person 4 (Rakesh)
    image_request = build_image_request(
        question=question,
        analysis=analysis,
        answer=final_answer,
    )

    # Attach extra metadata to the final answer dict
    final_answer["analysis"] = analysis
    final_answer["image_request"] = image_request

    return final_answer
