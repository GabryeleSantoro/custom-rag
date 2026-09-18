"""In-memory fixture store.

Stands in for SQLite plus LanceDB. Every method here maps to something the real
store will do; nothing above this layer knows the data is fake.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ragcore.api.schemas import (
    AppSettings,
    ChatMessage,
    ChatProject,
    ChatSession,
    Connection,
    Document,
    DocumentChunkRef,
    DocumentContent,
    DocumentPage,
    EvalSet,
    IndexStats,
    InstalledModel,
    Source,
    SourceCreate,
)
from ragcore.config import Config
from ragcore.stub.corpus import LoadedDoc, find_fixture_dir, load_corpus
from ragcore.stub.retrieval import Retriever

SCHEMA_VERSION = 1
EMBED_MODEL = "Qwen3-Embedding-0.6B-Q8_0"
EMBED_DIM = 1024
RERANK_MODEL = "Qwen3-Reranker-0.6B-Q8_0"


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class Store:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.started_at = _now()

        self.sources: dict[str, Source] = {}
        self.documents: dict[str, Document] = {}
        self.loaded: dict[str, LoadedDoc] = {}
        self.connections: dict[str, Connection] = {}
        self.models: dict[str, InstalledModel] = {}
        self.sessions: dict[str, ChatSession] = {}
        self.projects: dict[str, ChatProject] = {}
        self.messages: dict[str, list[ChatMessage]] = {}
        self.retriever = Retriever([])

        self.settings = AppSettings(
            onboarded=True,
            storage_path=str(config.data_dir),
            active_connection_id=None,
        )
        self.eval_sets = [
            EvalSet(name="base", n_questions=50, description="Public corpus, committed"),
            EvalSet(name="private", n_questions=32, description="Personal documents, local only"),
        ]

        self._seed()

    # ------------------------------------------------------------------ seeding

    def _seed(self) -> None:
        self._seed_models()
        self._seed_connections()
        fixture_dir = find_fixture_dir()
        if fixture_dir:
            source = self.add_source(
                SourceCreate(path=str(fixture_dir), include_globs=["**/*.md", "**/*.txt"])
            )
            self.ingest_source(source.id)
        self._seed_sessions()

    def _seed_models(self) -> None:
        shipped = [
            InstalledModel(
                id="embed-qwen3-0.6b",
                role="embedding",
                name="Qwen3 Embedding 0.6B",
                repo_id="Qwen/Qwen3-Embedding-0.6B-GGUF",
                filename="Qwen3-Embedding-0.6B-Q8_0.gguf",
                quant="Q8_0",
                size_bytes=639_000_000,
                sha256="a" * 64,
                active=True,
                shipped=True,
                downloaded_at=_now() - timedelta(days=2),
                context_length=32768,
            ),
            InstalledModel(
                id="rerank-qwen3-0.6b",
                role="reranking",
                name="Qwen3 Reranker 0.6B",
                repo_id="Qwen/Qwen3-Reranker-0.6B-GGUF",
                filename="Qwen3-Reranker-0.6B-Q8_0.gguf",
                quant="Q8_0",
                size_bytes=639_000_000,
                sha256="b" * 64,
                active=True,
                shipped=True,
                downloaded_at=_now() - timedelta(days=2),
                context_length=32768,
            ),
        ]
        for model in shipped:
            self.models[model.id] = model

    def _seed_connections(self) -> None:
        conn = Connection(
            id="conn_lmstudio",
            name="LM Studio",
            kind="openai-compatible",
            base_url="http://localhost:1234/v1",
            model_id="qwen3-8b-instruct",
            context_window=32768,
            max_output_tokens=2048,
            thinking="off",
            is_remote=False,
            has_api_key=False,
            active=True,
            created_at=_now() - timedelta(days=1),
        )
        self.connections[conn.id] = conn
        self.settings.active_connection_id = conn.id

    def _seed_sessions(self) -> None:
        for title, ago in (("Reranking candidate counts", 2), ("How chunking works", 26)):
            session = ChatSession(
                id=_id("chat"),
                title=title,
                message_count=0,
                created_at=_now() - timedelta(hours=ago),
                updated_at=_now() - timedelta(hours=ago),
            )
            self.sessions[session.id] = session
            self.messages[session.id] = []

    # ------------------------------------------------------------------ sources

    def add_source(self, payload: SourceCreate) -> Source:
        source = Source(id=_id("src"), added_at=_now(), **payload.model_dump())
        self.sources[source.id] = source
        return source

    def remove_source(self, source_id: str) -> int:
        self.sources.pop(source_id, None)
        removed = [d for d in self.documents.values() if d.source_id == source_id]
        for doc in removed:
            self.documents.pop(doc.id, None)
            self.loaded.pop(doc.id, None)
        self.rebuild_index()
        return len(removed)

    def ingest_source(self, source_id: str) -> list[Document]:
        """Load every file under a source and mark it indexed."""
        source = self.sources.get(source_id)
        if source is None:
            return []
        directory = Path(source.path)
        if not directory.is_dir():
            return []

        docs: list[Document] = []
        for loaded in load_corpus(
            directory,
            include_globs=source.include_globs,
            exclude_globs=source.exclude_globs,
            max_file_mb=source.max_file_mb,
        ):
            document = Document(
                id=loaded.doc_id,
                source_id=source_id,
                path=str(loaded.path),
                title=loaded.title,
                ext=loaded.ext,
                mime=loaded.mime,
                size_bytes=loaded.size_bytes,
                n_pages=len(loaded.pages),
                n_chunks=len(loaded.chunks),
                lang="en",
                status="indexed",
                keywords=loaded.keywords,
                mtime=loaded.mtime,
                indexed_at=_now(),
            )
            self.documents[document.id] = document
            self.loaded[document.id] = loaded
            docs.append(document)

        owned = [d for d in self.documents.values() if d.source_id == source_id]
        source.document_count = len(owned)
        source.indexed_count = len([d for d in owned if d.status == "indexed"])
        source.last_scan_at = _now()
        self.rebuild_index()
        return docs

    def rebuild_index(self) -> None:
        chunks = [c for doc in self.loaded.values() for c in doc.chunks]
        self.retriever = Retriever(chunks)

    # ---------------------------------------------------------------- documents

    def doc_meta(self) -> dict[str, dict]:
        return {
            d.id: {"source_id": d.source_id, "ext": d.ext, "lang": d.lang, "mtime": d.mtime}
            for d in self.documents.values()
        }

    def content(self, doc_id: str) -> DocumentContent | None:
        loaded = self.loaded.get(doc_id)
        document = self.documents.get(doc_id)
        if loaded is None or document is None:
            return None
        page_text = {p.page: p.text for p in loaded.pages}
        return DocumentContent(
            doc_id=doc_id,
            title=document.title,
            path=document.path,
            n_pages=len(loaded.pages),
            pages=[
                DocumentPage(page=p.page, section_path=p.section_path, text=p.text)
                for p in loaded.pages
            ],
            chunks=[
                DocumentChunkRef(
                    chunk_id=c.chunk_id,
                    page=c.page,
                    char_start=c.char_start,
                    char_end=c.char_end,
                    # Sliced from the page rather than copied off the chunk: the
                    # reader highlights by offset, so the two have to agree
                    # character for character.
                    text=page_text.get(c.page, "")[c.char_start : c.char_end],
                )
                for c in loaded.chunks
            ],
        )

    # ------------------------------------------------------------------- stats

    def index_stats(self) -> IndexStats:
        chunks = sum(len(d.chunks) for d in self.loaded.values())
        return IndexStats(
            documents=len(self.documents),
            chunks=chunks,
            parents=sum(len(d.pages) for d in self.loaded.values()),
            topics=0,
            size_bytes=sum(d.size_bytes for d in self.documents.values()),
            embed_model=EMBED_MODEL,
            embed_dim=EMBED_DIM,
            reranker_model=RERANK_MODEL,
            schema_version=SCHEMA_VERSION,
            last_indexed_at=max(
                (d.indexed_at for d in self.documents.values() if d.indexed_at), default=None
            ),
        )

    # ------------------------------------------------------------------- chats

    def create_project(self, name: str) -> ChatProject:
        now = _now()
        project = ChatProject(
            id=_id("project"),
            name=name.strip() or "Untitled project",
            created_at=now,
            updated_at=now,
        )
        self.projects[project.id] = project
        return project

    def create_session(
        self,
        title: str | None,
        scope_doc_id: str | None = None,
        project_id: str | None = None,
    ) -> ChatSession:
        now = _now()
        session = ChatSession(
            id=_id("chat"),
            title=title or "New chat",
            message_count=0,
            scope_doc_id=scope_doc_id,
            project_id=project_id,
            created_at=now,
            updated_at=now,
        )
        self.sessions[session.id] = session
        self.messages[session.id] = []
        return session

    def append_message(self, message: ChatMessage) -> None:
        self.messages.setdefault(message.session_id, []).append(message)
        session = self.sessions.get(message.session_id)
        if session:
            session.message_count = len(self.messages[message.session_id])
            session.updated_at = message.created_at
            if session.title == "New chat" and message.role == "user":
                session.title = message.text[:48].strip() or session.title

    def new_id(self, prefix: str) -> str:
        return _id(prefix)
