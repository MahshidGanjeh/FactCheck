### Line-by-line explanation for `farsi_factcheck/agent.py`

1. `import os` — Access environment variables and paths
2. `import re` — Regular expressions for text normalization/splitting
3. `import json` — Parse JSON from trafilatura extractor
4. `import time` — Reserved for potential timing; currently unused
5. `from dataclasses import dataclass` — Compact data containers for results
6. `from typing import Any, Dict, List, Optional, Tuple` — Type annotations
7. `import torch` — Backend library required by Transformers models
8. `from duckduckgo_search import DDGS` — Web search without API keys
9. `from langdetect import detect` — Language identification (we keep Persian)
10. `from tqdm import tqdm` — Progress bars during scoring
11. `import trafilatura` — Robust article text extraction
12. `from transformers import pipeline` — Load NLI model as a pipeline
13. `DEFAULT_NLI_MODEL_ID = ...` — Default multilingual NLI model (mDeBERTa-v3-base XNLI); can be overridden via env var
14. `@dataclass class EvidenceSentence` — Holds an evidence sentence with URL/title and NLI scores
15. `@dataclass class FactCheckResult` — Final output with scores and selected evidence
16. `def is_url(...)` — Simple pattern check to decide if input is a URL
17. `def normalize_persian(...)` — Normalizes common Arabic/Persian character variants and whitespace
18. `def simple_sentence_split(...)` — Splits text into sentences using Persian/Latin punctuation
19. `def tokenize_words(...)` — Basic tokenization after normalization and punctuation removal
20. `def jaccard_overlap(...)` — Measures lexical overlap between claim and sentence
21. `def extract_text_from_url(...)` — Downloads URL and extracts title/text using trafilatura (JSON first, fallback to plain text)
22. `def extract_claim_from_text(...)` — Picks a short, informative candidate (title or leading sentences) as the claim
23. `class FactChecker.__init__(...)` — Initializes the NLI pipeline; auto-selects GPU if available
24. `FactChecker._search(...)` — Uses DuckDuckGo to get candidate pages in region Iran
25. `FactChecker._filter_and_fetch(...)` — Deduplicates, downloads, filters by language, keeps Persian docs
26. `FactChecker._select_relevant_sentences(...)` — Ranks sentences by lexical overlap and selects top-k per doc
27. `FactChecker._nli_scores(...)` — Computes entailment/contradiction/neutral probabilities for (premise, hypothesis)
28. `FactChecker.check(...)` — Main flow: parse input, search, fetch docs, select sentences, score with NLI, aggregate to a validity percentage and return top evidence

### Line-by-line explanation for `farsi_factcheck/__main__.py`

1. `import argparse` — Parse CLI arguments
2. `import json` — Print JSON result
3. `from typing import Optional` — Type hints
4. `from .agent import FactChecker` — Import the agent
5. `def main():` — CLI entry point
6. `ArgumentParser(...)` — CLI description
7. `--input` — Required: sentence or URL
8. `--max-results` — Search results to fetch
9. `--max-docs` — Max documents to download
10. `--top-k-sentences` — Evidence sentences per document
11. `--model-id` — Optional override for the NLI model id
12. `args = parser.parse_args()` — Read CLI args
13. `checker = FactChecker(...)` — Create agent
14. `result = checker.check(...)` — Run fact-checking
15. `print(json.dumps(...))` — Emit structured JSON (UTF-8 intact)
16. `if __name__ == "__main__": main()` — Standard Python CLI pattern

For any specific line you want expanded further, please refer to the inline code and the numbered list above.