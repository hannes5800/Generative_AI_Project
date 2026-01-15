# Project Structure

```text
project_root/
│
├── app_demo_screenshots
│    └── comparison.png       # Comparison question asked with image generation in the app
│    └── simple.png           # Simple question asked with image generation in the app
├── data/
│   └── raw_pdfs/             # Uploaded/downloaded crypto whitepapers (PDFs)
├── src/
|   ├── static/
│   |   └── style.css         # Frontend styling (dark-mode UI)
|   ├── templates/
│   |   └── index.html        # Main frontend template (Q&A UI, toggles, upload form)
|   ├── app.py                # Flask web app (API endpoints, orchestration, PDF upload handling, RAG initialization)
│   ├── corpus.py             # PDF loading, text extraction, and cleaning
|   ├── imaging.py            # Image prompt construction and image generation
|   ├── pipeline.py           # LLM pipeline (question analysis, answer generation)
│   └── rag.py                # Chunking, embeddings, Qdrant retrieval logic
├── Crypto_Demo.ipynb         # Demo notebook explaining core mechanics step-by-step
├── Crypto_Demo_Image.png     # Example generated image (fallback if notebook image expires)
├── OpenAI_API.txt            # OpenAI API key (required for image generation, here just a placeholder)
└── README.md                 # Project documentation (this file)
