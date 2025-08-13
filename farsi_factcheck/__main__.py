import argparse
import json
from typing import Optional

from .agent import FactChecker


def main():
    parser = argparse.ArgumentParser(
        description="Persian (Farsi) fact-checking agent using web evidence and multilingual NLI"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="A Persian sentence (claim) or a URL to a news article",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=8,
        help="Max number of search results to retrieve",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=6,
        help="Max number of documents to download and analyze",
    )
    parser.add_argument(
        "--top-k-sentences",
        type=int,
        default=3,
        help="How many evidence sentences to select per document",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Optional Hugging Face model id for NLI (defaults to XLM-RoBERTa XNLI)",
    )

    args = parser.parse_args()

    checker = FactChecker(nli_model_id=args.model_id or None)
    result = checker.check(
        user_input=args.input,
        max_results=args.max_results,
        max_docs=args.max_docs,
        top_k_sentences=args.top_k_sentences,
    )

    print(
        json.dumps(
            {
                "claim": result.claim,
                "validity_percent": result.validity_percent,
                "support_score": result.support_score,
                "refute_score": result.refute_score,
                "neutral_score": result.neutral_score,
                "evidence": [
                    {
                        "sentence": e.sentence,
                        "url": e.url,
                        "title": e.title,
                        "lexical_overlap": e.lexical_overlap,
                        "entailment": e.entailment,
                        "contradiction": e.contradiction,
                    }
                    for e in result.evidence
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()