import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag import _ollama_post, chunk_text, connect_db, delete_document, extract_pages, list_documents


class ChunkTextTests(unittest.TestCase):
    def test_chunks_cover_text_and_overlap(self):
        text = "one two three four five six seven eight nine ten"
        chunks = chunk_text(text, max_chars=16, overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 16 for chunk in chunks))
        self.assertIn("one", chunks[0])
        self.assertIn("ten", chunks[-1])
        self.assertTrue(any(set(chunks[index].split()) & set(chunks[index + 1].split()) for index in range(len(chunks) - 1)))

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("text", max_chars=10, overlap=10)


class LocalStorageTests(unittest.TestCase):
    def test_database_does_not_change_existing_directory_permissions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            directory.chmod(0o755)
            connection = connect_db(directory / "medical_history.db")
            connection.close()

            self.assertEqual(directory.stat().st_mode & 0o777, 0o755)

    def test_text_upload_extraction_and_source_deletion(self):
        self.assertEqual(extract_pages("note.txt", b"Family history"), [("text", "Family history")])
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "medical_history.db"
            connection = connect_db(database)
            with connection:
                connection.execute(
                    "INSERT INTO documents (source_hash, member, filename, added_at) VALUES (?, ?, ?, ?)",
                    ("test-hash", "Alex", "note.txt", "2026-01-01T00:00:00+00:00"),
                )
            connection.close()

            self.assertEqual(list_documents(database)[0]["filename"], "note.txt")
            delete_document(database, "test-hash")
            self.assertEqual(list_documents(database), [])

    def test_remote_ollama_endpoint_is_rejected(self):
        with patch("rag.OLLAMA_HOST", "https://example.com"):
            with self.assertRaisesRegex(RuntimeError, "local loopback"):
                _ollama_post("/api/embed", {})


if __name__ == "__main__":
    unittest.main()