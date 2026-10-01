# Family Medical History RAG

A local-first archive for searching family medical records. Uploaded PDFs and text files are parsed on this computer, embedded by a local Ollama model, and stored as text and vectors in a local SQLite database. Questions are answered by a local Ollama chat model and accompanied by source excerpts.

This is a personal information-retrieval tool, not a medical device. It can miss or misstate information; verify every result against the cited document and consult a qualified clinician for medical decisions.

## Requirements

- Python 3.11 or newer
- [Ollama](https://ollama.com/) installed and running locally
- Enough memory for the selected local models (the defaults are small models)

The app defaults to `llama3.2:3b` for answers and `nomic-embed-text` for embeddings. Model downloads happen through Ollama; family documents and questions are sent only to the Ollama service on this computer.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
streamlit run app.py
```

If Ollama is not running as a background service, start it in another terminal with `ollama serve`. The app binds to `127.0.0.1:8501`.

Optional environment variables:

- `OLLAMA_HOST`: local Ollama URL (default `http://127.0.0.1:11434`)
- `CHAT_MODEL`: local answer model (default `llama3.2:3b`)
- `EMBEDDING_MODEL`: local embedding model (default `nomic-embed-text`)
- `RAG_DB_PATH`: local SQLite database path (default `data/medical_history.db`)

## Use

Upload text-based PDFs, `.txt`, or `.md` files and assign each upload to a family member. Choose a person (or all family members) before asking a question. Each answer includes retrieved source excerpts; PDF citations include page numbers. You can remove indexed sources from the sidebar.

Scanned PDFs are not OCR'd in this first version. No original files are retained by the app, but extracted medical text and embeddings remain in the local database until their source is removed. Keep the database on an encrypted, access-controlled device and do not commit it or share it. Deleting a source in the app deletes its indexed chunks; it does not securely erase prior backups or storage-device copies.

Run the local, data-free tests with:

```bash
python -m unittest discover -s tests
```