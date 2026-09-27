# Chat with Your Notes

A local-first retrieval-augmented generation (RAG) app. Upload PDFs, Markdown, or text files and ask questions whose answers are grounded in the uploaded material.

The vector database is local and in memory. The app still sends document text to the OpenAI Embeddings API and sends retrieved excerpts to the OpenAI chat model, so it is **not an offline or fully local AI app**.

## What this project demonstrates

- Document ingestion with LangChain's `PyPDFLoader`
- Chunking with `RecursiveCharacterTextSplitter`
- Semantic search with `OpenAIEmbeddings` and an in-memory Chroma collection
- Conversational retrieval with `create_retrieval_chain`
- Grounded answers, refusal behavior, and visible source excerpts
- Stateful chat and document indexing in Streamlit
- Prompt-injection resistance: retrieved text is explicitly treated as data, not instructions
- Public-demo safeguards for upload size, page count, prompt length, output length, retries, and per-session usage

## Architecture

```text
PDF / TXT / MD
      │
      ▼
 PyPDFLoader / text decoder
      │
      ▼
 RecursiveCharacterTextSplitter
      │
      ▼
 OpenAIEmbeddings ──► in-memory Chroma
                           │
Question ──► history-aware retriever ──► relevant chunks
                                             │
                                             ▼
                                  grounded answer + sources
```

## Run locally

Python 3.10+ is recommended. This project has also been verified with Python 3.14.

```bash
cd chat-with-your-notes
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
# Put your OpenAI API key in .env, then:
streamlit run app.py
```

When running locally, `.env` pre-fills the password field. You can also leave `.env` unset and paste a key into the sidebar for the current session. Never commit `.env` or a real API key.

## Publish it on Streamlit Community Cloud

Yes, this project can be shared through a public `streamlit.app` URL:

1. Create a new GitHub repository and push this project to it.
2. Open [Streamlit Community Cloud](https://share.streamlit.io), choose **Create app**, select the repository, and set `app.py` as the entrypoint.
3. In **Advanced settings**, select Python 3.12. Do **not** configure an OpenAI secret for the hosted app.
4. Deploy, test by entering your own API key in the password field, and add the resulting URL to your GitHub README and portfolio.

The public demo uses a **bring-your-own-key** model. Every visitor supplies their own OpenAI API key and pays for their own API usage. The app keeps the value in the current Streamlit session and passes it directly to the LangChain OpenAI clients; this code does not write it to disk, logs, environment variables, URLs, or browser storage.

However, a hosted Streamlit application is still server-side software: the key passes through the machine running the app and exists in that process's memory while requests run. Visitors should only enter a short-lived, project-scoped key into a deployment they trust. Running a clone locally with `.env` provides the strongest trust boundary.

### Public-demo limits

The app currently enforces:

- 5 files per index, 10 MB each, and 25 MB combined
- 120 pages per PDF and 600,000 extracted characters per index
- 750 chunks per index
- 3 indexing operations and 20 questions per browser session
- 1,000 characters per question, 700 output tokens, 30-second API timeouts, and 2 retries
- Only the latest 12 chat messages are sent as conversational history

The browser-session counters are soft controls for responsiveness rather than billing protection—a visitor can refresh to obtain a new session. Because the app requires each visitor's own key, usage is charged to that visitor's OpenAI account. Users should apply appropriate limits to the OpenAI Project associated with the key they enter.

## Test and lint

```bash
pytest
ruff check .
```

## How RAG works here

1. **Load:** PDFs become one LangChain `Document` per page. Text and Markdown files become one `Document` each.
2. **Split:** Long documents are divided into overlapping chunks. Overlap reduces the chance that an important sentence is cut away from its context.
3. **Embed:** `text-embedding-3-small` converts every chunk into a vector that captures semantic meaning.
4. **Store:** Chroma keeps those vectors in memory for the current Streamlit session.
5. **Retrieve:** A user question is embedded, then Chroma returns the nearest chunks.
6. **Generate:** The chat model receives only the question, chat history, and retrieved chunks. The prompt requires it to refuse unsupported answers.

RAG reduces hallucinations but cannot guarantee factual answers. Retrieval can miss the right passage, and a model can misread retrieved text. The source panel makes the evidence easy to inspect.

### Supported-document boundaries

- Works best with text-based PDFs, Markdown, and UTF-8 text files.
- Image-only/scanned PDFs are detected when no readable text is extracted and require OCR before upload.
- Complex tables, diagrams, handwriting, and multi-column layouts may not extract perfectly with `PyPDFLoader`.
- Ordinary questions, extraction, comparison, and passage-level summaries work through retrieval. A reliable summary of an entire long document needs a separate map-reduce summarization pipeline; the prompt deliberately avoids claiming full coverage when only retrieved excerpts are available.

## Portfolio talking points

- Why chunk size and overlap affect recall, cost, and answer quality
- Why “Chroma is local” does not mean the whole pipeline is private
- How chat-history rewriting improves follow-up questions such as “What happened next?”
- Why displaying retrieved evidence is more trustworthy than returning an answer alone
- How you would evaluate it with a small question/answer dataset using retrieval recall and answer faithfulness

## Suggested next iterations

1. Add a RAG evaluation set and automated quality report.
2. Add scanned-PDF OCR (the current version detects image-only PDFs and explains the limitation).
3. Persist Chroma collections per user and deduplicate chunks by content hash.
4. Add hybrid keyword + vector retrieval and reranking.
5. Add authentication and a persistent, server-side rate limiter for a broadly promoted demo.
