import re
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct 
import uuid

# Creating chunks for the text
def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """
    Splits a text into overlapping chunks.

    Args:
        text (str): The text that will be split into chunks.
        chunk_size (int): Maximum number of words per chunk.
        overlap (int): Number of words that overlap between chunks.

    Returns:
        list[str]: List of chunks.
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

def create_chunk_objects(
    docs: list[dict]
) -> list[dict]:
    """
    Converts a list of documents into structured overlapping text chunks for RAG systems. Uses the chunk_text function.

    Args:
        docs (list[dict]): List of documents,
            - 'project_id' (str): Clear identifier for the project.
            - 'source_path' (str): Path to the original document.
            - 'text' (str): Full corpus of the document.

    Returns:
        list[dict]: List of chunk dictionaries:
            - 'id' (str): Unique chunk ID, e.g. 'bitcoin_0'.
            - 'project_id' (str): Project to which the chunk belongs to.
            - 'source' (str): Original document path.
            - 'chunk_index' (int): Position of the chunk in the document.
            - 'text' (str): Text corpus of the chunk.
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

def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    Generates embeddings for a list of chunks using a sentence transformer.

    Args:
        chunks (list[dict]): List of chunk dictionaries with:
            - 'text' (str): Text corpus of the chunk.

    Returns:
        np.ndarray: Array of embeddings for each chunk, so it can be passed to an vector databases.
    """

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # Extracts the text dictionary
    texts = [c["text"] for c in chunks]

    # Using the sentence encoder for all texts to create embeddings
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    return embeddings


def init_qdrant_collection(embeddings: np.ndarray, COLLECTION: str = "crypto_whitepapers") -> None:
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

    # Creates or overwrites a collcetion
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=embeddings.shape[1], # Dimensionality of the vectors == embeddings
            distance=Distance.COSINE # # Similarity metric
        )
    )


    return client, COLLECTION

def upload_to_qdrant(client: QdrantClient, chunks, embeddings: np.ndarray, collection: str) -> None:
    """
    Uploads the chunks and their embeddings to the specified Qdrant collection.

    Args:
        client (QdrantClient): Pre-defined Qdrant client.
        chunks (list[dict]): List of chunk dictionaries:
            - 'chunk_index' (int): Position of the chunk in the document.
            - 'project_id' (str): Identifier of the project.
            - 'source' (str): Original document path.
            - 'text' (str): Text content of the chunk.
        embeddings (np.ndarray): Embeddings for each each chunk.

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


def retrieve_rag(question: str, client: QdrantClient, COLLECTION: str, top_k: int = 5) -> list[dict]:

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
        output (list[dict]): List of chunk dictionaries:
            - 'text' (str): Chunk text
            - 'project' (str): Project ID
            - 'doc_id' (str): Original document ID/path
            - 'chunk_id' (str): Unique chunk ID
            - 'score' (float): Similarity score
    """

    # Creating embeddings the question using the SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    question_embedding = model.encode(question).tolist()  # List for Qdrant

    # Searching the vector database collection for top_k similar chunks
    results = client.query_points(
        collection_name=COLLECTION,
        query=question_embedding,
        limit=top_k,
        with_payload=True
    )

    # Result list
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