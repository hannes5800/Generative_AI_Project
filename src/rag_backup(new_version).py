"""
RAG Module - Retrieval-Augmented Generation Components
=======================================================

This module handles:
- Text chunking with overlap for better retrieval
- Embedding generation using sentence transformers  
- Qdrant vector database initialization and upload
- Semantic search with optional project filtering
"""

import re
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue


def chunk_text(
    text: str, 
    chunk_size: int = 600, 
    overlap: int = 100
) -> list[str]:
    """
    Splits a text into overlapping chunks.

    Args:
        - text: the text that will be split into chunks
        - chunk_size: maximum number of words per chunk
        - overlap: number of words that overlap between chunks

    Returns:
        - chunks: list of chunks
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words): 
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks


def create_chunk_objects(docs: list[dict]) -> list[dict]:
    """
    Converts a list of documents into structured overlapping text chunks.

    Args:
        docs: list of document dicts with:
            - "project_id": identifier for the project
            - "source_path": path to the original document
            - "text": full text of the document

    Returns:
        List of chunk dictionaries with id, project_id, source, chunk_index, text
    """
    all_chunks = []

    for doc in docs:
        project_id = doc["project_id"]
        source = doc["source_path"]
        full_text = doc["text"]
        chunks = chunk_text(full_text)

        for idx, chunk in enumerate(chunks):
            chunk_obj = {
                "id": f"{project_id}_{idx}",
                "project_id": project_id,
                "source": source,
                "chunk_index": idx,
                "text": chunk
            }
            all_chunks.append(chunk_obj)

    return all_chunks


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    Generates embeddings for a list of chunks using a sentence transformer.

    Args:
        chunks: list of chunk dictionaries with 'text' field

    Returns:
        np.ndarray: array of embeddings for each chunk
    """
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def init_qdrant_collection(
    embeddings: np.ndarray, 
    collection_name: str = "crypto_whitepapers"
) -> tuple[QdrantClient, str]:
    """
    Initialize a Qdrant collection in-memory.

    Args:
        - embeddings: needed to determine vector size
        - collection_name: name of the collection

    Returns:
        - client: Qdrant client
        - collection_name: name of the collection
    """
    client = QdrantClient(":memory:")

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embeddings.shape[1],
            distance=Distance.COSINE
        )
    )

    return client, collection_name


def upload_to_qdrant(
    client: QdrantClient, 
    chunks: list[dict], 
    embeddings: np.ndarray, 
    collection_name: str
) -> None:
    """
    Uploads chunks and embeddings to Qdrant collection.

    Args:
        - client: Qdrant client
        - chunks: list of chunk dictionaries
        - embeddings: embeddings for each chunk
        - collection_name: name of the collection
    """
    points = []
    
    for chunk, emb in zip(chunks, embeddings):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={
                "chunk_id": f"{chunk['project_id']}_{chunk['chunk_index']}",
                "project_id": chunk["project_id"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"]
            }
        ))

    client.upsert(collection_name, points)
    print(f"Uploaded {len(points)} chunks.")


def retrieve_rag(
    question: str, 
    client: QdrantClient, 
    collection_name: str, 
    top_k: int = 5,
    project_filter: list[str] | None = None
) -> list[dict]:
    """
    Retrieve relevant chunks from Qdrant using semantic search.
    
    For comparison queries (2+ projects), chunks are retrieved separately
    for each project to ensure balanced representation.
    
    Args:
        - question: the user's query
        - client: Qdrant client instance
        - collection_name: name of the Qdrant collection
        - top_k: maximum number of chunks to return
        - project_filter: optional list of project IDs to filter by

    Returns:
        list[dict]: List of chunk dictionaries with text, project, score, etc.
    """
    # Create embedding for the question
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    question_embedding = model.encode(question).tolist()

    # --- Handle comparison queries (multiple projects) ---
    # Retrieve chunks separately for each project to ensure balanced coverage
    if project_filter and len(project_filter) > 1:
        chunks_per_project = max(2, top_k // len(project_filter))
        
        all_results = []
        for project in project_filter:
            # Filter for this specific project
            single_filter = Filter(
                must=[
                    FieldCondition(
                        key="project_id",
                        match=MatchValue(value=project)
                    )
                ]
            )
            
            results = client.query_points(
                collection_name=collection_name,
                query=question_embedding,
                query_filter=single_filter,
                limit=chunks_per_project,
                with_payload=True
            )
            
            for point in results.points:
                payload = point.payload
                all_results.append({
                    "text": payload.get("text", ""),
                    "project": payload.get("project_id", ""),
                    "doc_id": payload.get("source", ""),
                    "chunk_id": payload.get("chunk_id", ""),
                    "score": point.score
                })
        
        # Sort by score and return top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    # --- Handle single project filter ---
    qdrant_filter = None
    if project_filter and len(project_filter) == 1:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="project_id",
                    match=MatchValue(value=project_filter[0])
                )
            ]
        )

    # --- Handle no filter (search all) ---
    results = client.query_points(
        collection_name=collection_name,
        query=question_embedding,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True
    )

    output = []
    for point in results.points:
        payload = point.payload
        output.append({
            "text": payload.get("text", ""),
            "project": payload.get("project_id", ""),
            "doc_id": payload.get("source", ""),
            "chunk_id": payload.get("chunk_id", ""),
            "score": point.score 
        })

    return output