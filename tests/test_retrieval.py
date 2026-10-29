"""Regresiones del rechazo incorrecto de «que es rag?».

Sin red: python -m unittest discover -s tests
Con el corpus y el modelo local: RUN_RAG_INTEGRATION=1 HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1 python -m unittest discover -s tests
"""
import os
import unittest

from rag.pipeline import Prepared, RAGPipeline
from rag.prompts import REFUSAL
from rag.retrieval import LexicalIndex, semantic_query
from rag.vectorstore import Hit


class RetrievalRegressionTests(unittest.TestCase):
    def test_acronym_query_is_stable_across_accents_case_and_punctuation(self):
        queries = ["que es rag?", "¿Qué es RAG?", "qué es rag", "QUE ES RAG???"]
        self.assertEqual(len({semantic_query(q) for q in queries}), 1)
        self.assertIn("Retrieval Augmented Generation", semantic_query(queries[0]))

    def test_unrelated_queries_are_not_rewritten(self):
        question = "¿Quién ganó el Mundial de fútbol de 2018?"
        self.assertEqual(semantic_query(question), question)

    def test_lexical_search_ignores_titles_and_bibliographies(self):
        index = LexicalIndex(
            ["title", "references", "concept"],
            ["Introducción a RAG", "[1] RAG in production, https://example.com/paper, pp. 10-20.\n"
             "[2] Another RAG study in arXiv, 2024, including retrieval augmented generation.",
             "RAG significa Retrieval Augmented Generation y combina recuperación de información con generación de texto."],
            [{}, {}, {}],
        )
        self.assertEqual(index.search("que es rag?"), ["concept"])

    def test_refusal_keeps_consulted_passages_available_for_inspection(self):
        hit = Hit(1, "Un pasaje recuperado para revisar.", "curso.pdf", 4, 3, .3)
        prepared = Prepared("pregunta", "pregunta", [hit], [hit], ["contexto"])
        answer = RAGPipeline.finish(prepared, REFUSAL)
        self.assertTrue(answer.refused)
        self.assertEqual(answer.hits, [hit])
        self.assertEqual(answer.cited, [])


@unittest.skipUnless(os.getenv("RUN_RAG_INTEGRATION") == "1", "requiere el corpus y el modelo local")
class CorpusRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = RAGPipeline()

    def test_exact_reported_question_retrieves_explanatory_course_material(self):
        for query in ["que es rag?", "¿Qué es RAG?", "QUE ES RAG???"]:
            with self.subTest(query=query):
                prepared = self.pipe.prepare(query)
                self.assertEqual(len(prepared.hits), 4)
                self.assertEqual(prepared.hits[0].source, "Introducción a RAG.pdf")
                self.assertEqual(prepared.hits[0].page, 4)
                self.assertIn("Nueva", prepared.hits[0].text)
                for hit in prepared.hits:
                    self.assertIn(hit.text, prepared.messages[-1].content)

    def test_selected_context_respects_requested_limit(self):
        for k in (1, 4, 8):
            with self.subTest(k=k):
                hits, _ = self.pipe.retrieve("que es rag?", k)
                self.assertGreater(len(hits), 0)
                self.assertLessEqual(len(hits), k)
                self.assertTrue(all(h.distance <= self.pipe.s.max_distance for h in hits))

    def test_unrelated_football_question_still_has_no_context(self):
        prepared = self.pipe.prepare("¿Quién ganó el Mundial de fútbol de 2018?")
        self.assertEqual(prepared.hits, [])
        self.assertIsNone(prepared.messages)


if __name__ == "__main__":
    unittest.main()
