# Project Structure

```text
project_root/
  data/
    raw_pdfs/            # Downloaded whitepapers
  src/
    corpus.py            # Loading and cleaning docs
    rag.py               # Chunking, embeddings, retrieval
    pipeline.py          # LLM pipeline (analysis ? answer ? review)
    imaging.py           # Image generation tool
  Crypto_Demo.ipynb      # Demo with explanations
  Crypto_Demo_Image.png  # Generated image as PNG (in case it's not visible in the demo notebook)
  OpenAI_API.txt         # txt file for the OpenAI API key (needed for image generation, currently the file contains a placeholder)