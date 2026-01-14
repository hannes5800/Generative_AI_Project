import re
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct 
import uuid


# Creating chunks for the text
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

    # Loop until the end of the text is reached     
    while start < len(words): 
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks


def create_chunk_objects(docs: list[dict]) -> list[dict]:
    """
    Converts a list of documents into structured overlapping text chunks for RAG systems. Uses the chunk_text function.

    Args:
        docs: list of document dicts with:
            - "project_id": clear identifier for the project
            - "source_path": path to the original document
            - "text": full corpus of the document

    Returns a list of chunk dictionaries with:
        - "id": Unique chunk ID, e.g. 'bitcoin_0'
        - "project_id": project to which the chunk belongs to
        - "source": original document path
        - "chunk_index": position of the chunk in the document
        - "text": text corpus of the chunk
    """
        
    all_chunks = []

    # Looping through each document in the input list
    for doc in docs:
        # Extracting the project ID, source path, and full text
        project_id = doc["project_id"]
        source = doc["source_path"]
        full_text = doc["text"]
        # Here the function chunk_text is used to split the full text into smaller overlapping chunks
        chunks = chunk_text(full_text)

        # Looping through each chunk and create a structured dictionary
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
        chunks: list of chunk dictionaries with 'text': text corpus of the chunk

    Returns:
        np.ndarray: array of embeddings for each chunk, so it can be passed to an vector databases
    """

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # Extracts the text dictionary
    texts = [c["text"] for c in chunks]

    # Using the sentence encoder for all texts to create embeddings
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    return embeddings


def init_qdrant_collection(embeddings: np.ndarray, collection_name: str = "crypto_whitepapers") -> None:
    """
    Initialize a Qdrant collection via in-memory

    Args:
        - embeddings: the embeddings are needed to set the vector size
        - collection_name: name of the collection in Qdrant

    Returns:
        - client: Qdrant client with the collection ready
        - collection_name: name of the collection
    """

    # Initializing an in-memory Qdrant client (could be replaced with docker)
    client = QdrantClient(":memory:")

    # Creates or overwrites a collcetion
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embeddings.shape[1],  # Dimensionality of the vectors == embeddings
            distance=Distance.COSINE   # Similarity metric
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
    Uploads the chunks and their embeddings to the specified Qdrant collection.

    Args:
        - client: pre-defined Qdrant client
        - chunks: list of chunk dictionaries:
        - embeddings: embeddings for each each chunk
        - collection_name: name of the collection

    Returns:
        - None
    """

    points = []
    
    # Loop through each chunk and its embedding
    for chunk, emb in zip(chunks, embeddings):
        points.append(PointStruct(
            id=str(uuid.uuid4()),  # Unique UUID for Qdrant
            vector=emb.tolist(),   # Embedding as list
            payload={
                "chunk_id": f"{chunk['project_id']}_{chunk['chunk_index']}",  # readable ID
                "project_id": chunk["project_id"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"]
            }
        ))

    # Uploading all points to the Qdrant collection
    client.upsert(collection_name, points)
    print(f"Uploaded {len(points)} chunks.")

# Updated with project filter, because majority of wrong chunks were used for some requests and random chunk ratios were pulled for comparisons (sometimes 1 to 0 ratio)
def retrieve_rag(
    question: str, 
    client: QdrantClient, 
    COLLECTION: str, 
    top_k: int = 5,
    project_filter: list[str] | None = None
) -> list[dict]:
    """
    Retrieve relevant chunks from Qdrant using semantic search.
    
    Args:
        - question: the user's query in plain text
        - client: Qdrant client instance
        - COLLECTION: name of the Qdrant collection
        - top_k: maximum number of chunks to return
        - project_filter: optional list of project IDs to filter by.
          If provided, only chunks from these projects are returned.
          For comparison queries (2+ projects), chunks are distributed
          evenly across projects to ensure balanced coverage.

    Returns:
        list[dict]: List of chunk dictionaries with text, project, score, etc.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    # Create embedding for the question
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    question_embedding = model.encode(question).tolist()

    # Handle comparison queries (multiple projects)
    # When comparing projects, we retrieve chunks separately for each project
    # to ensure balanced representation. Otherwise one project might dominate.
    if project_filter and len(project_filter) > 1:
        # Calculate chunks per project (distribute evenly)
        chunks_per_project = max(1, top_k // len(project_filter))
        
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
            
            # Retrieve top chunks for this project
            results = client.query_points(
                collection_name=COLLECTION,
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
        
        # Sort by score (highest first) and return
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    # Handle single project or no filter
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

    # Search with optional filter
    results = client.query_points(
        collection_name=COLLECTION,
        query=question_embedding,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True
    )

    output = []
    for point in results.points:
        payload = point.payload
        chunk_dict = {
            "text": payload.get("text", ""),
            "project": payload.get("project_id", ""),
            "doc_id": payload.get("source", ""),
            "chunk_id": payload.get("chunk_id", ""),
            "score": point.score 
        }
        output.append(chunk_dict)

    return output
