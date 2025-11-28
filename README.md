# Project Structure

```text
project_root/
  data/
    raw_pdfs/       # downloaded whitepapers
    texts/          # extracted & cleaned text
  src/
    corpus.py       # loading and cleaning docs -> Person 1
    indexer.py      # chunking + embeddings + retrieval -> Person 2
    pipeline.py     # LLM pipeline (analysis ? answer ? review) -> Person 3
    fine_tuning.py  # Optional: fine-tuning logic (train + load) -> Person 3
    imaging.py      # image generation tool -> Person 4
  main.ipynb        # final demo + explanations