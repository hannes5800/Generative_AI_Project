project_root/
  data/
  |  raw_pdfs/      # downloaded whitepapers
  |  texts/         # extracted text
  src/
  |  corpus.py      # loading and cleaning docs
  |  indexer.py     # chunking + embeddings + retrieval
  |  pipeline.py    # LLM pipeline (analysis ? answer ? review)
  |  imaging.py     # image generation tool
  main.ipynb        # final demo + explanations