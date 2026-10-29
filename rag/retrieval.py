"""Selección híbrida: significado, términos exactos y fragmentos con contenido útil."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter


STOP_WORDS = set("""
a al algo algunas algunos ante antes como con cual cuales cuando de del desde donde el ella
ellos en entre era es esa esas ese eso esos esta estas este esto estos ha hay la las le les lo
los mas me mi mis muy no nos o para pero por porque que quien se si sin sobre son su sus te
tiene tienen tu tus un una unas uno unos y ya the a an and are as at be by for from how in is
it of on or that the this to was what which with
""".split())


def normalize(text: str) -> str:
    folded = "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))
    folded = re.sub(r"\btf\s*[-_ ]?\s*idf\b", "tfidf", folded)
    folded = re.sub(r"\bbm\s*25\b", "bm25", folded)
    return " ".join(re.findall(r"[a-z0-9]+", folded))


def semantic_query(question: str) -> str:
    """Las siglas y variantes sin tildes deben buscar el mismo concepto, no portadas."""
    query = normalize(question)
    expansions = {
        "rag": "RAG Retrieval Augmented Generation",
        "tfidf": "TF-IDF term frequency inverse document frequency",
        "bm25": "BM25 Best Matching 25",
        "llm": "LLM large language model",
        "llms": "LLMs large language models",
    }
    if not any(term in query.split() for term in expansions):
        return question
    for term, expansion in expansions.items():
        query = re.sub(rf"\b{term}\b", expansion, query)
    return query


def has_content(text: str) -> bool:
    """No usar portadas, títulos aislados ni listas bibliográficas como explicaciones."""
    if len(normalize(text).split()) < 12:
        return False
    references = re.findall(r"(?m)^\s*\[\d+\]", text)
    bibliography = any(marker in text.lower() for marker in ("arxiv", "https://", "proceedings", "pp."))
    return not (len(references) >= 2 and bibliography)


class LexicalIndex:
    """BM25 sobre el corpus; sin servicios externos ni dependencias adicionales."""

    def __init__(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        self.records = {key: (text, meta) for key, text, meta in zip(ids, documents, metadatas) if has_content(text)}
        self.tokens = {key: Counter(t for t in normalize(text).split() if t not in STOP_WORDS)
                       for key, (text, _) in self.records.items()}
        self.lengths = {key: sum(tokens.values()) for key, tokens in self.tokens.items()}
        self.average_length = sum(self.lengths.values()) / max(len(self.lengths), 1)
        self.frequencies = Counter(token for tokens in self.tokens.values() for token in tokens)

    def search(self, question: str) -> list[str]:
        terms = set(normalize(question).split()) - STOP_WORDS
        scores = []
        for key, tokens in self.tokens.items():
            score = 0.0
            for term in terms:
                frequency = tokens[term]
                if not frequency:
                    continue
                df = self.frequencies[term]
                idf = math.log(1 + (len(self.tokens) - df + .5) / (df + .5))
                norm = 1.5 * (.25 + .75 * self.lengths[key] / max(self.average_length, 1))
                score += idf * frequency * 2.5 / (frequency + norm)
            if score:
                # Una coincidencia parcial no debe desplazar un pasaje que cubre toda la consulta.
                coverage = sum(term in tokens for term in terms) / max(len(terms), 1)
                scores.append((score * coverage, key))
        return [key for _, key in sorted(scores, key=lambda item: (-item[0], item[1]))]
