from dataclasses import dataclass

import pytest
from langchain_core.documents import Document

from note_chat.ingestion import MAX_FILES, fingerprint_files, load_uploaded_files, split_documents
from note_chat.rag import MAX_QUESTION_CHARS, RagSession, ask, source_label


@dataclass
class FakeUpload:
    name: str
    content: bytes

    def getvalue(self) -> bytes:
        return self.content


def test_fingerprint_is_order_independent() -> None:
    first = FakeUpload("a.txt", b"alpha")
    second = FakeUpload("b.md", b"beta")
    assert fingerprint_files([first, second]) == fingerprint_files([second, first])


def test_loads_text_and_preserves_source() -> None:
    documents = load_uploaded_files([FakeUpload("notes.txt", b"Hello notes")])
    assert documents[0].page_content == "Hello notes"
    assert documents[0].metadata == {
        "source": "notes.txt",
        "page": 0,
        "citation": "notes.txt",
    }


def test_rejects_unknown_extension() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        load_uploaded_files([FakeUpload("notes.csv", b"a,b")])


def test_rejects_too_many_files() -> None:
    files = [FakeUpload(f"{index}.txt", b"notes") for index in range(MAX_FILES + 1)]
    with pytest.raises(ValueError, match="at most"):
        load_uploaded_files(files)


def test_split_requires_overlap_smaller_than_chunk() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_documents([Document(page_content="Some text")], 100, 100)


def test_source_label_uses_human_page_number() -> None:
    document = Document(page_content="x", metadata={"source": "guide.pdf", "page": 2})
    assert source_label(document) == "guide.pdf, page 3"


def test_rejects_overlong_question_before_calling_chain() -> None:
    session = RagSession(chain=None, vector_store=None, chunk_count=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="limited"):
        ask(session, "x" * (MAX_QUESTION_CHARS + 1), [])
