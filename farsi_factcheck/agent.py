import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from duckduckgo_search import DDGS
from langdetect import detect
from tqdm import tqdm
import trafilatura
from transformers import pipeline


DEFAULT_NLI_MODEL_ID = os.environ.get(
    "HF_FA_NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
)


@dataclass
class EvidenceSentence:
    sentence: str
    url: str
    title: str
    lexical_overlap: float
    entailment: float
    contradiction: float


@dataclass
class FactCheckResult:
    claim: str
    validity_percent: float
    support_score: float
    refute_score: float
    neutral_score: float
    evidence: List[EvidenceSentence]


def is_url(text: str) -> bool:
    return bool(re.match(r"^https?://", text.strip()))


def normalize_persian(text: str) -> str:
    if not text:
        return ""
    # Normalize common Arabic/Persian variants and whitespace
    mapping = {
        "ي": "ی",
        "ك": "ک",
        "ۀ": "ه",
        "ؤ": "و",
        "أ": "ا",
        "إ": "ا",
        "‌": " ",  # ZWNJ to space for simpler matching
        "\u200c": " ",  # explicit ZWNJ
    }
    for src, dst in mapping.items():
        text = text.replace(src, dst)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def simple_sentence_split(text: str) -> List[str]:
    # Split on Persian/Arabic/Latin sentence boundaries
    parts = re.split(r"(?<=[\.\!\؟\?\n])\s+", text)
    sentences: List[str] = []
    for p in parts:
        s = p.strip()
        if 5 <= len(s) <= 600:
            sentences.append(s)
    return sentences


def tokenize_words(text: str) -> List[str]:
    text = normalize_persian(text)
    # Remove punctuation (keep Persian letters and digits)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [t for t in text.split(" ") if t]


def jaccard_overlap(a_tokens: List[str], b_tokens: List[str]) -> float:
    set_a = set(a_tokens)
    set_b = set(b_tokens)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def extract_text_from_url(url: str) -> Tuple[str, str]:
    downloaded = trafilatura.fetch_url(url, no_ssl=True)
    if not downloaded:
        return "", ""
    data_json = trafilatura.extract(
        downloaded,
        output="json",
        include_links=False,
        favor_recall=True,
        include_formatting=False,
    )
    if data_json:
        try:
            data = json.loads(data_json)
            text = data.get("text") or ""
            title = data.get("title") or ""
            return text, title
        except Exception:
            pass
    # Fallback to plain text extract
    text = trafilatura.extract(
        downloaded,
        output="txt",
        include_links=False,
        favor_recall=True,
        include_formatting=False,
    )
    return text or "", ""


def extract_claim_from_text(text: str, title: str = "") -> str:
    if title and len(title) > 10:
        title_clean = normalize_persian(title)
    else:
        title_clean = ""
    text_clean = normalize_persian(text)
    candidates: List[str] = []
    if title_clean:
        candidates.append(title_clean)
    candidates.extend(simple_sentence_split(text_clean)[:5])
    # Choose the first reasonably informative candidate
    for cand in candidates:
        num_words = len(tokenize_words(cand))
        if num_words >= 6:
            return cand
    # Fallback
    return candidates[0] if candidates else text_clean[:200]


class FactChecker:
    def __init__(
        self,
        nli_model_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        model_id = nli_model_id or DEFAULT_NLI_MODEL_ID
        self.model_id = model_id
        self.nli_pipeline = pipeline(
            "text-classification",
            model=model_id,
            tokenizer=model_id,
            device=0 if self.device == "cuda" else -1,
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )

    def _search(self, query: str, max_results: int = 8) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        with DDGS() as ddgs:
            for r in ddgs.text(
                query,
                region="ir-ir",
                safesearch="moderate",
                timelimit=None,
                max_results=max_results,
            ):
                results.append(r)
        return results

    def _filter_and_fetch(self, results: List[Dict[str, Any]], max_docs: int = 6) -> List[Tuple[str, str, str]]:
        cleaned: List[Tuple[str, str, str]] = []  # (url, title, text)
        seen_urls: set[str] = set()
        for r in results:
            url = r.get("href") or r.get("link") or r.get("url")
            title = r.get("title") or r.get("body") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            try:
                text, extracted_title = extract_text_from_url(url)
            except Exception:
                text, extracted_title = "", ""
            if not text or len(text) < 300:
                continue
            # Language filter: prefer Persian
            sample = normalize_persian(text[:400])
            lang_ok = False
            try:
                lang_ok = detect(sample) == "fa"
            except Exception:
                lang_ok = False
            if not lang_ok:
                continue
            final_title = normalize_persian(extracted_title or title)
            cleaned.append((url, final_title, normalize_persian(text)))
            if len(cleaned) >= max_docs:
                break
        return cleaned

    def _select_relevant_sentences(
        self, claim: str, doc_text: str, url: str, title: str, top_k: int = 3
    ) -> List[EvidenceSentence]:
        claim_tokens = tokenize_words(claim)
        sentences = simple_sentence_split(doc_text)
        scored: List[Tuple[float, str]] = []
        for s in sentences:
            overlap = jaccard_overlap(claim_tokens, tokenize_words(s))
            if overlap > 0:
                scored.append((overlap, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected: List[EvidenceSentence] = []
        for overlap, s in scored[:top_k]:
            selected.append(
                EvidenceSentence(
                    sentence=s,
                    url=url,
                    title=title,
                    lexical_overlap=overlap,
                    entailment=0.0,
                    contradiction=0.0,
                )
            )
        return selected

    def _nli_scores(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        outputs = self.nli_pipeline({"text": premise, "text_pair": hypothesis})
        # outputs is a list of dicts: [{"label": "CONTRADICTION", "score": ...}, ...]
        label_to_score = {d["label"].upper(): d["score"] for d in outputs[0]}
        entail = label_to_score.get("ENTAILMENT", 0.0)
        contradict = label_to_score.get("CONTRADICTION", 0.0)
        neutral = label_to_score.get("NEUTRAL", 0.0)
        return entail, contradict, neutral

    def check(
        self,
        user_input: str,
        max_results: int = 8,
        max_docs: int = 6,
        top_k_sentences: int = 3,
    ) -> FactCheckResult:
        if is_url(user_input):
            text, title = extract_text_from_url(user_input)
            claim = extract_claim_from_text(text, title)
        else:
            claim = normalize_persian(user_input)
        if not claim or len(claim) < 6:
            raise ValueError("Claim text is too short to fact-check.")

        search_results = self._search(claim, max_results=max_results)
        docs = self._filter_and_fetch(search_results, max_docs=max_docs)

        evidence: List[EvidenceSentence] = []
        for url, title, text in docs:
            evidence.extend(
                self._select_relevant_sentences(
                    claim, text, url, title, top_k=top_k_sentences
                )
            )

        support_scores: List[float] = []
        refute_scores: List[float] = []
        neutral_scores: List[float] = []
        weights: List[float] = []

        for ev in tqdm(evidence, desc="Scoring evidence", leave=False):
            try:
                ent, con, neu = self._nli_scores(ev.sentence, claim)
            except Exception:
                ent, con, neu = 0.0, 0.0, 0.0
            ev.entailment = ent
            ev.contradiction = con
            support_scores.append(ent)
            refute_scores.append(con)
            neutral_scores.append(neu)
            weights.append(max(1e-4, ev.lexical_overlap))

        def weighted_average(values: List[float], w: List[float]) -> float:
            if not values or not w:
                return 0.0
            s_w = sum(w)
            if s_w <= 0:
                return 0.0
            return sum(v * wi for v, wi in zip(values, w)) / s_w

        support = weighted_average(support_scores, weights)
        refute = weighted_average(refute_scores, weights)
        neutral = weighted_average(neutral_scores, weights)

        denom = max(1e-6, support + refute)
        validity = support / denom
        validity_percent = float(round(100.0 * validity, 2))

        # Keep top 5 evidence sentences by max(support, refute) * weight
        evidence.sort(
            key=lambda e: max(e.entailment, e.contradiction) * (e.lexical_overlap + 1e-4),
            reverse=True,
        )
        top_evidence = evidence[:5]

        return FactCheckResult(
            claim=claim,
            validity_percent=validity_percent,
            support_score=support,
            refute_score=refute,
            neutral_score=neutral,
            evidence=top_evidence,
        )