from typing import List, Dict
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, SearchParams
import uuid

# Creating chunks for the text
def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """
    Splits a text into overlapping chunks.

    Args:
        text (str): The text to split into chunks.
        chunk_size (int): Maximum number of words per chunk.
        overlap (int): Number of words that overlap between chunks.

    Returns:
        list[str]: List of text chunks.
    """

    words = text.split()
    chunks = []
    start = 0

    # Loop until we reach the end of the text     
    while start < len(words): 
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

def create_chunk_objects(docs: List[Dict]) -> List[Dict]:
    """
    Converts a list of documents into structured overlapping text chunks for RAG systems.

    Args:
        docs (List[Dict]): List of documents, each containing:
            - 'project_id' (str): Identifier for the project.
            - 'source_path' (str): Path to the original document.
            - 'text' (str): Full textual content of the document.

    Returns:
        List[Dict]: List of chunk dictionaries, each containing:
            - 'id' (str): Unique chunk ID, e.g., 'bitcoin_0'.
            - 'project_id' (str): Project this chunk belongs to.
            - 'source' (str): Original document path.
            - 'chunk_index' (int): Position of the chunk in the document.
            - 'text' (str): Text content of the chunk.
    """
        
    all_chunks = []

    # Looping through each document in the input list
    for doc in docs:
        # Extracting the project ID, source path, and full text
        project_id = doc["project_id"]
        source = doc["source_path"]
        full_text = doc["text"]
        # Here we use the chunk_text function to split the full text into smaller overlapping chunks
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

def embed_chunks(chunks: List[Dict]):
    """
    Generates embeddings for a list of text chunks using a sentence transformer.

    Args:
        chunks (List[Dict]): List of chunk dictionaries, each with:
            - 'text' (str): Text content of the chunk.

    Returns:
        np.ndarray: Array of embeddings for each chunk, so it can be passed to an vector databases.
    """

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # Extracts the text dictionary
    texts = [c["text"] for c in chunks]
    # Using the sentence encoder for all texts to create embeddings
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def init_qdrant_collection(embeddings, collection_name: str = "crypto_whitepapers"):
    """
    Initialize a Qdrant collection via in-memory

    Args:
        embeddings (np.ndarray): The embeddings are needed to set the vector size.
        collection_name (str): Name of the collection in Qdrant.

    Returns:
        client (QdrantClient): Qdrant client with the collection ready.
        collection_name (str): Name of the collection.
    """

    # Initializing an in-memory Qdrant client / could be replaced with docker
    client = QdrantClient(":memory:")

    # Name of the collection
    COLLECTION = "crypto_whitepapers"

    # Creates or overwrites a collcetion
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=embeddings.shape[1], # Dimensionality of the vectors == embeddings
            distance=Distance.COSINE # # Similarity metric
        )
    )


    return client, COLLECTION

def upload_to_qdrant(client, chunks, embeddings, collection):
    """
    Uploads text chunks and their embeddings to the specified Qdrant collection.

    Args:
        client (QdrantClient): Pre-defined Qdrant client.
        chunks (List[Dict]): List of chunk dictionaries:
            - 'chunk_index' (int): Position of the chunk in the document.
            - 'project_id' (str): Identifier of the project.
            - 'source' (str): Original document path.
            - 'text' (str): Text content of the chunk.
        embeddings (np.ndarray): Embeddings corresponding to each chunk.

    Returns:
        None
    """

    points = []
    
    # Loop through each chunk and its embedding
    for chunk, emb in zip(chunks, embeddings):
        points.append(PointStruct(
            id=str(uuid.uuid4()), # Unique UUID for Qdrant
            vector=emb.tolist(), # Embedding as list
            payload={
                "chunk_id": f"{chunk['project_id']}_{chunk['chunk_index']}",  # lesbare ID
                "project_id": chunk["project_id"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"]
            }
        ))

    # Uploading all points to the Qdrant collection
    client.upsert(collection, points)
    print(f"Uploaded {len(points)} chunks.")


def query_rag_flexible(question: str, top_k: int = 5) -> list[dict]:
    """
    Flexible retrieval of relevant chunks from Qdrant using semantic search.

    Steps:
    1. User question gets embedded.
    2. Query Qdrant retreives the top k chunks.
    3. Each chunk gets formatted as a dictionary with text, project_id, doc_id, and chunk_id.
    4. Count project hits for relevance scoring.

    Args:
        question (str): The user's query in plain text.
        top_k (int): Maximum number of chunks to return.

    Returns:
        output (list[dict]): List of chunk dictionaries, each with keys:
            - 'text' (str): Chunk text
            - 'project' (str): Project ID
            - 'doc_id' (str): Original document ID/path
            - 'chunk_id' (str): Unique chunk ID
        project_hits (dict): Dictionary with project_id as key and number of hits as value
    """

    # Embeds the question using the SentenceTransformer
    question_embedding = model.encode(question).tolist()  # Liste für Qdrant

    # Searching the vector database collection for top_k similar chunks
    results = client.query_points(
        collection_name=COLLECTION,
        query=question_embedding,
        limit=top_k
    )

    # Result list
    output = []
    project_hits = {}
    
    for point in results.points:
        payload = point.payload
        chunk_dict = {
            "text": payload.get("text", ""),
            "project": payload.get("project_id", ""),
            "doc_id": payload.get("source", ""),
            "chunk_id": payload.get("chunk_id", ""),
        }
        output.append(chunk_dict)

        # Projekt-Häufigkeiten zählen
        pid = payload.get("project_id")
        if pid:
            project_hits[pid] = project_hits.get(pid, 0) + 1

    return output, project_hits