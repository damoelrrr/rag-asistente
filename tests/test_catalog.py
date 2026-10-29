"""El alcance del tutor proviene del catálogo, no de ejemplos sueltos del curso."""
import unittest
from unittest.mock import Mock

from rag.catalog import catalog_answer, is_catalog_question
from rag.pipeline import RAGPipeline
from rag.vectorstore import Hit, VectorStore


class CatalogueTests(unittest.TestCase):
    def test_scope_questions_are_distinguished_from_subject_matter(self):
        for question in (
            "sobre que temas puedes hablar?", "¿Sobre qué temas podemos hablar?",
            "¿En qué me puedes ayudar?", "¿Qué documentos tienes cargados?",
            "¿Cuáles son tus fuentes?", "Lista tus fuentes disponibles",
            "¿Qué temas conoces?", "Por favor, dime sobre qué temas puedes hablar",
        ):
            with self.subTest(question=question):
                self.assertTrue(is_catalog_question(question))
        for question in (
            "¿Qué es el tema en un prompt?", "¿Qué temas debo incluir en un prompt?",
            "¿Qué son las fuentes en RAG?", "¿Qué es RAG?", "¿Qué sabes sobre BM25?",
            "Explícame el ejemplo de energías renovables del documento",
        ):
            with self.subTest(question=question):
                self.assertFalse(is_catalog_question(question))

    def test_scope_answer_uses_indexed_documents_even_with_misleading_history(self):
        pipe = RAGPipeline.__new__(RAGPipeline)
        pipe.store = Mock()
        pipe.store.catalogue.return_value = [
            Hit(1, "Una explicación del curso.", "Introducción a RAG.pdf", 4, 3, None),
            Hit(2, 'Ejemplos: escribir sobre energías renovables o adoptar una mascota.',
                "Introducción_Ingeniería_Prompt (3).pdf", 3, 5, None),
        ]
        pipe._llm = Mock()
        pipe.rewrite = Mock(side_effect=AssertionError("No reescribir preguntas sobre el alcance"))
        pipe.retrieve = Mock(side_effect=AssertionError("No confundir catálogo con similitud de fragmentos"))
        history = [{'role': 'assistant', 'content': 'Puedo hablar de historia y adoptar mascotas.'}]
        answer = pipe.answer("sobre que temas puedes hablar?", history, top_k=1)
        self.assertIn("Introducción a RAG [1]", answer.text)
        self.assertIn("ingeniería de prompts [2]", answer.text)
        self.assertNotIn("energías renovables", answer.text)
        self.assertNotIn("adoptar una mascota", answer.text)
        self.assertEqual(answer.cited, [1, 2])
        self.assertEqual(len(answer.hits), 2)
        self.assertFalse(answer.refused)
        pipe._llm.invoke.assert_not_called()
        prepared = pipe.prepare("¿Cuáles son tus fuentes?", history)
        self.assertEqual("".join(pipe.stream(prepared)), prepared.direct_answer)

    def test_catalogue_reports_only_files_present_in_the_index(self):
        store = VectorStore.__new__(VectorStore)
        store.col = Mock()
        passage = "RAG incorpora información externa relevante para mejorar las respuestas que genera un modelo de lenguaje."
        store.col.get.return_value = {
            'documents': ['Introducción', passage],
            'metadatas': [
                {'source': 'mis_notas.md', 'page': -1, 'chunk_index': 0},
                {'source': 'mis_notas.md', 'page': -1, 'chunk_index': 1},
            ],
        }
        catalogue = store.catalogue()
        self.assertEqual(len(catalogue), 1)
        self.assertEqual(catalogue[0].text, passage)
        self.assertEqual(catalogue[0].source, 'mis_notas.md')
        self.assertIsNone(catalogue[0].similarity)
        answer = catalog_answer(catalogue)
        self.assertIn('mis notas [1]', answer)
        self.assertNotIn('Ollama', answer)

    def test_empty_index_does_not_invent_course_topics(self):
        self.assertIn('Todavía no hay documentos indexados', catalog_answer([]))
        self.assertNotIn('BM25', catalog_answer([]))


if __name__ == '__main__':
    unittest.main()
