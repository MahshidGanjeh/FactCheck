## Persian Fact-Checking Agent (Farsi)

This repository contains a simple Persian (Farsi) fact-checking agent that:

- Searches the web (DuckDuckGo) for evidence related to a claim
- Downloads and extracts article text (trafilatura)
- Uses a multilingual NLI model from Hugging Face to estimate whether evidence supports or contradicts the claim
- Returns a validity percentage and top evidence snippets

### Setup

1) Create a virtual environment (recommended) and install dependencies:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) (Optional) If you have a GPU and PyTorch CUDA available, the model will use it automatically.

### Run

- Check a Persian sentence (claim):

```bash
python -m farsi_factcheck --input "این خبر می‌گوید واکسن فلان باعث نازایی می‌شود"
```

- Check a news article by URL (the agent extracts a candidate claim from the page):

```bash
python -m farsi_factcheck --input "https://example.com/persian-news-article"
```

- Optional: choose a different NLI model id (must support XNLI labels such as XLM-R/XNLI, mDeBERTa XNLI):

```bash
python -m farsi_factcheck --input "..." --model-id "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
```

The output is JSON and includes `validity_percent` and selected evidence.

### Notes

- This is a heuristic system, not a definitive fact-checker. It uses lexical overlap to select evidence sentences and NLI to estimate support/refute.
- By default it uses `joeddav/xlm-roberta-large-xnli`, which is accurate but large. You can try a smaller multilingual NLI model if resource constrained.