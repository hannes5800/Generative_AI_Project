"""
Flask Web Application for Crypto Whitepaper Q&A System
=======================================================

This module serves as the main entry point for the web application.
It connects the RAG (Retrieval-Augmented Generation) pipeline with
a simple web interface, allowing users to ask questions about 
cryptocurrency whitepapers through their browser.

Features:
- Ask questions about uploaded whitepapers
- Upload new PDF whitepapers via the web interface
- Automatic chunking and embedding of new documents

Architecture Overview:
- Flask handles HTTP routing and serves the frontend
- On startup, the RAG system is initialized (loading docs, creating embeddings)
- User questions are processed via POST endpoint that returns JSON responses
- New PDFs can be uploaded and added to the vector database dynamically

Dependencies:
- Flask: Web framework for routing and templating
- corpus.py: PDF extraction and text cleaning
- rag.py: Chunking, embedding, and vector search
- pipeline.py: Question analysis and answer generation via LLM
"""

import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from corpus import load_corpus, extract_pdf, clean_text
from rag import (
    create_chunk_objects, 
    embed_chunks, 
    init_qdrant_collection, 
    upload_to_qdrant, 
    retrieve_rag
)
from pipeline import analyze_question, generate_answer

# Initialize Flask app
# __name__ tells Flask where to look for templates and static files
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the data directory where PDFs are stored
# Using ../data because app.py is in src/ folder
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Subdirectory within data/ where all PDFs are stored
# This keeps all whitepapers organized in one place: data/raw_pdfs/
PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {"pdf"}

# Maximum file size (16 MB)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
# We store the Qdrant client and chunks globally so they persist across requests.
# In a production setting, you might use a proper database connection pool
# or a persistent Qdrant instance (Docker) instead of in-memory storage.

client = None          # Will hold the Qdrant client after initialization
all_chunks = []        # Store all chunks for potential re-indexing
all_embeddings = None  # Store embeddings array
COLLECTION = "crypto_whitepapers"  # Name of our vector collection


def allowed_file(filename: str) -> bool:
    """
    Check if a filename has an allowed extension.
    
    Args:
        filename: The name of the uploaded file
        
    Returns:
        True if the file extension is in ALLOWED_EXTENSIONS
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def init_rag():
    """
    Initialize the complete RAG pipeline on application startup.
    
    This function performs the following steps:
    1. Load all PDF documents from the data/ directory
    2. Split documents into overlapping chunks for better retrieval
    3. Generate vector embeddings for each chunk using a sentence transformer
    4. Create an in-memory Qdrant collection to store the vectors
    5. Upload all chunks with their embeddings to Qdrant
    
    Note: This runs once when the server starts. For large document collections,
    consider pre-computing embeddings and persisting them to disk.
    
    The 'global' statements allow us to modify module-level variables.
    """
    global client, all_chunks, all_embeddings
    
    # Step 1: Load documents
    # load_corpus() reads all PDFs from data/raw_pdfs/ and extracts clean text
    print("Loading corpus from data/raw_pdfs/ directory...")
    docs = load_corpus(PDF_DIR)
    print(f"  → Loaded {len(docs)} documents")
    
    # Step 2: Create chunks
    # Chunking splits long documents into smaller pieces (600 words with 100 overlap)
    # This improves retrieval accuracy since we can find specific relevant passages
    print("Creating text chunks...")
    all_chunks = create_chunk_objects(docs)
    print(f"  → Created {len(all_chunks)} chunks")
    
    # Step 3: Generate embeddings
    # Each chunk is converted to a 384-dimensional vector using BAAI/bge-small-en-v1.5
    # These vectors capture semantic meaning, enabling similarity search
    print("Generating embeddings (this may take a moment)...")
    all_embeddings = embed_chunks(all_chunks)
    print(f"  → Generated embeddings with shape {all_embeddings.shape}")
    
    # Step 4: Initialize vector database
    # Qdrant stores vectors and allows fast similarity search
    # Using in-memory mode here; for persistence, use Qdrant with Docker
    print("Initializing Qdrant collection...")
    client, _ = init_qdrant_collection(all_embeddings, COLLECTION)
    
    # Step 5: Upload to database
    # Each chunk is stored with its embedding and metadata (project, source, etc.)
    print("Uploading chunks to Qdrant...")
    upload_to_qdrant(client, all_chunks, all_embeddings, COLLECTION)
    
    print("=" * 50)
    print("RAG system ready! Server starting...")
    print("=" * 50)


def get_available_projects() -> list[str]:
    """
    Scan the raw_pdfs directory and return a list of available project names.
    
    Projects are identified by PDF filenames (without extension) in data/raw_pdfs/.
    For example, if data/raw_pdfs/ contains:
        data/raw_pdfs/bitcoin.pdf
        data/raw_pdfs/ethereum.pdf
    
    This returns: ["bitcoin", "ethereum"]
    
    Returns:
        List of project names (PDF filenames without .pdf extension)
    """
    projects = []
    if os.path.exists(PDF_DIR):
        for item in os.listdir(PDF_DIR):
            # Only include PDF files
            if item.lower().endswith(".pdf"):
                # Remove .pdf extension to get project name
                project_name = item.rsplit(".", 1)[0]
                projects.append(project_name)
    return sorted(projects)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """
    Serve the main page of the application.
    
    Passes the list of available projects to the template for the dropdown menu.
    
    Returns:
        HTML content of the main page
    """
    projects = get_available_projects()
    return render_template("index.html", projects=projects)


@app.route("/ask", methods=["POST"])
def ask():
    """
    Process a user question and return an AI-generated answer.
    
    This endpoint:
    1. Receives a JSON payload with the user's question
    2. Analyzes the question to detect mentioned projects and question type
    3. Retrieves the most relevant chunks from the vector database
    4. Sends the question + context to the LLM for answer generation
    5. Returns the answer with citations as JSON
    
    Request format:
        POST /ask
        Content-Type: application/json
        {"question": "What is Bitcoin's consensus mechanism?"}
    
    Response format:
        {
            "answer": "Bitcoin uses Proof of Work...",
            "citations": [{"chunk_number": 1, "project": "bitcoin", ...}],
            "analysis": {"projects": ["bitcoin"], "type": "overview", ...}
        }
    """
    # Parse JSON body from the request
    data = request.get_json()
    question = data.get("question", "")
    
    # Validate input - reject empty questions
    if not question.strip():
        return jsonify({"error": "No question provided"}), 400
    
    # Step 1: Analyze the question
    # This detects which projects are mentioned (bitcoin, ethereum, etc.)
    # and classifies the question type (overview, comparison, risk, tokenomics)
    # 
    # We pass the dynamically detected projects from our PDF folder
    # so any newly uploaded project is automatically recognized
    available = get_available_projects()
    analysis = analyze_question(question, available_projects=available)
    
    # Step 2: Retrieve relevant chunks
    # The question is embedded and compared against all chunk embeddings
    # Returns the top 5 most semantically similar chunks
    # 
    # NEW: If specific projects are detected in the question, we filter
    # the retrieval to only include chunks from those projects.
    # This prevents cross-contamination (e.g., Ethereum chunks for Solana questions)
    detected_projects = analysis.get("projects", [])
    
    chunks = retrieve_rag(
        question, 
        client, 
        COLLECTION, 
        top_k=5,
        project_filter=detected_projects if detected_projects else None
    )
    
    # Step 3: Generate answer
    # The LLM receives the question + retrieved chunks as context
    # It produces an answer grounded in the provided information
    result = generate_answer(question, chunks)
    
    # Return structured JSON response
    return jsonify({
        "answer": result["answer_text"],
        "citations": result["citations"],
        "analysis": analysis
    })


@app.route("/upload", methods=["POST"])
def upload_pdf():
    """
    Handle PDF upload and add to the RAG system.
    
    This endpoint:
    1. Receives a PDF file and project name from a form submission
    2. Validates the file (must be PDF, within size limit)
    3. Saves the file to data/{project_name}/
    4. Extracts text, creates chunks, generates embeddings
    5. Adds the new chunks to the existing Qdrant collection
    
    Request format:
        POST /upload
        Content-Type: multipart/form-data
        - file: PDF file
        - project_name: Name for the project (creates subdirectory)
    
    Response format:
        {"success": true, "message": "...", "chunks_added": 15}
        or
        {"error": "..."}, 400
    """
    global client, all_chunks, all_embeddings
    
    # --- Validate Request ---
    
    # Check if file was included in request
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    project_name = request.form.get("project_name", "").strip().lower()
    
    # Check if a file was actually selected
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Check if project name was provided
    if not project_name:
        return jsonify({"error": "Project name is required"}), 400
    
    # Sanitize project name (remove special characters, spaces -> underscores)
    project_name = "".join(c if c.isalnum() or c == "_" else "_" for c in project_name)
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400
    
    # --- Save File ---
    
    # Save directly to raw_pdfs/ folder (no subdirectories)
    # The project_name becomes the filename: data/raw_pdfs/{project_name}.pdf
    filename = f"{project_name}.pdf"
    file_path = os.path.join(PDF_DIR, filename)
    
    # Check if a file with this project name already exists
    if os.path.exists(file_path):
        return jsonify({"error": f"Project '{project_name}' already exists"}), 400
    
    # Save the uploaded file
    file.save(file_path)
    print(f"Saved uploaded file to: {file_path}")
    
    # --- Process and Index ---
    
    try:
        # Extract text from the PDF
        raw_text = extract_pdf(file_path)
        cleaned_text = clean_text(raw_text)
        
        # Create document dict (same format as load_corpus returns)
        doc = {
            "document_class": project_name,
            "project_id": project_name,  # Use project_name as ID
            "text": cleaned_text,
            "source_path": file_path
        }
        
        # Create chunks for this document
        new_chunks = create_chunk_objects([doc])
        print(f"Created {len(new_chunks)} chunks from uploaded PDF")
        
        # Generate embeddings for new chunks
        new_embeddings = embed_chunks(new_chunks)
        
        # Upload to existing Qdrant collection
        # This adds to the existing vectors, doesn't replace them
        upload_to_qdrant(client, new_chunks, new_embeddings, COLLECTION)
        
        # Update global state
        all_chunks.extend(new_chunks)
        
        return jsonify({
            "success": True,
            "message": f"Successfully uploaded {filename} to project '{project_name}'",
            "chunks_added": len(new_chunks),
            "project_name": project_name
        })
        
    except Exception as e:
        # If processing fails, try to clean up the saved file
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500


@app.route("/projects", methods=["GET"])
def list_projects():
    """
    Return a list of all available projects.
    
    Useful for refreshing the dropdown after uploading a new project.
    
    Response format:
        {"projects": ["bitcoin", "ethereum", "solana", ...]}
    """
    projects = get_available_projects()
    return jsonify({"projects": projects})


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialize RAG system before starting the server
    # This loads all documents and prepares the vector database
    init_rag()
    
    # Start Flask development server
    # - debug=True: Auto-reload on code changes, detailed error pages
    # - port=5000: Access via http://localhost:5000
    # 
    # WARNING: debug=True should be disabled in production!
    # For production, use a proper WSGI server like gunicorn:
    #   gunicorn -w 4 app:app
    app.run(debug=True, port=5000)