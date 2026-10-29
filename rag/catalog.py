"""Preguntas sobre el alcance del tutor y títulos del catálogo de documentos."""
from __future__ import annotations

import re
from pathlib import Path

from .retrieval import normalize
from .vectorstore import Hit


DOCUMENT_TITLES = {
    "2312.10997v5.pdf": "Retrieval-Augmented Generation for LLMs",
    "3 BM25_Score.pdf": "BM25 · Relevancia en la búsqueda",
    "Ejecucin_de_LLMs_con_Ollama.pdf": "LLMs locales con Ollama",
    "El_Mtodo_Socrtico_El_Arte_de_Preguntar (3).pdf": "El método socrático · El arte de preguntar",
    "Introducción a RAG.pdf": "Introducción a RAG",
    "Introducción_Ingeniería_Prompt (3).pdf": "Introducción a la ingeniería de prompts",
    "TF-IDF_Keyword_search.pdf": "TF-IDF · Búsqueda por palabras clave",
}


def is_catalog_question(question: str) -> bool:
    """Distingue las capacidades del tutor del concepto «tema» dentro de un prompt."""
    text = normalize(question)
    text = re.sub(r"^(?:hola |por favor |dime |dime por favor |me puedes decir |puedes decirme |podrias decirme |quiero saber )+", "", text)
    patterns = (
        r"(?:sobre |de )?que (?:tipo de )?(?:temas?|contenidos?|materias?|asuntos?) "
        r"(?:me )?(?:puedes|podrias|podemos|sabes|puede el tutor) (?:hablar|tratar|explicar|ensenar|responder|ayudar)(?:me)?",
        r"que (?:temas|contenidos|materias) (?:conoces|manejas|dominas|cubres|explicas)",
        r"(?:sobre |de )?que puedes (?:hablar|tratar|ensenarme|explicarme)",
        r"(?:en |con )?que(?: temas)? me puedes ayudar",
        r"que (?:puedes hacer|sabes|sabes hacer)",
        r"(?:que|cuales|cuales son (?:las|tus|los)) (?:fuentes|documentos|archivos|materiales) "
        r"(?:tienes|usas|utilizas|consultas|conoces)(?: (?:disponibles|cargados|cargadas|indexados|indexadas))?",
        r"cuales son (?:tus|las) (?:fuentes|capacidades|tematicas)",
        r"cuales son (?:los|tus) (?:temas|documentos|archivos)(?: (?:disponibles|del curso|de la asignatura|indexados|cargados))?",
        r"(?:que|cuales) (?:temas|documentos|fuentes|archivos) "
        r"(?:hay|estan disponibles|estan cargados|estan indexados)(?: en (?:la biblioteca|el curso|la base de conocimientos))?",
        r"(?:sobre |de )?que (?:tratan|hablan) (?:tus|los) (?:documentos|fuentes|archivos)",
        r"(?:cual es |que incluye |muestrame |lista )?(?:el |tu )?(?:temario|catalogo de documentos|alcance del tutor|base de conocimientos)",
        r"(?:lista|enumera|muestrame) (?:tus|los|las) (?:fuentes|documentos|temas)(?: disponibles| del curso)?",
    )
    return any(re.fullmatch(pattern, text) for pattern in patterns)


def catalog_answer(documents: list[Hit]) -> str:
    if not documents:
        return "Todavía no hay documentos indexados para definir los temas del tutor. ¿Qué material del curso quieres utilizar?"
    lines = ["Puedo ayudarte con los temas del material del curso que está indexado en tu biblioteca:", ""]
    for document in documents:
        title = DOCUMENT_TITLES.get(document.source, Path(document.source).stem.replace("_", " "))
        title = re.sub(r"([\\`*_{}\[\]<>])", r"\\\1", title)
        lines.append(f"- {title} [{document.n}].")
    lines += ["", "Mi alcance se limita a ese material; los ejemplos usados dentro de un documento no amplían el temario.",
              "", "¿Cuál de estos temas te gustaría explorar primero?"]
    return "\n".join(lines)
