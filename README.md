# Project Structure

```text
project_root/
│
├── data/
│   └── raw_pdfs/                # Uploaded/downloaded crypto whitepapers (PDFs)
│
├── src/
│   ├── corpus.py                # PDF loading, text extraction, and cleaning
│   ├── rag.py                   # Chunking, embeddings, Qdrant retrieval logic
│   ├── pipeline.py              # LLM pipeline (question analysis, answer generation,
│   │                             # optional review pass, image request construction)
│   └── imaging.py               # Image prompt construction and image generation
│
├── static/
│   └── style.css                # Frontend styling (dark-mode UI)
│
├── templates/
│   └── index.html               # Main frontend template (Q&A UI, toggles, upload form)
│
├── app.py                       # Flask web application (API endpoints, orchestration,
│                                 # PDF upload handling, RAG initialization)
│
├── Crypto_Demo.ipynb             # Demo notebook explaining core mechanics step-by-step
├── Crypto_Demo_Image.png         # Example generated image (fallback if notebook image expires)
│
├── OpenAI_API.txt                # OpenAI API key (required for image generation;
│                                 # placeholder included by default)
│
└── README.md                     # Project documentation (this file)
