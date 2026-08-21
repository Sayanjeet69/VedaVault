"""Small HTTP adapter around the frozen VedaVault RAG V1 service."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vedavault_retrieval import (
    AnswerMode,
    ClarificationRequiredError,
    ConversationContext,
    ConversationStore,
    EvidenceHygieneError,
    GroqClient,
    GroqLLMProvider,
    GroqQueryUnderstandingProvider,
    IndexCompatibilityError,
    IndexManifestError,
    InMemoryConversationStore,
    LLMProviderError,
    LanguagePolicy,
    LocalVectorStore,
    QueryUnderstandingProviderError,
    Retriever,
    SentenceTransformerEmbeddingProvider,
    SupportedLanguage,
    VedaVaultResponse,
    VedaVaultService,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_PATH = (
    ROOT / "Data" / "Processed" / "Bhagavad_Gita" / "retrieval_index"
)
DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://localhost:4200",
    "http://localhost:5173",
)
V1_LANGUAGES = frozenset(
    {
        SupportedLanguage.ENGLISH,
        SupportedLanguage.HINDI,
        SupportedLanguage.BENGALI,
        SupportedLanguage.SANSKRIT,
    }
)


class AnswerRequest(BaseModel):
    query: str
    input_language: SupportedLanguage | None = None
    response_language: SupportedLanguage | None = None
    mode: AnswerMode = AnswerMode.TEXTUAL
    session_id: str | None = None


class ScripturalTeachingResponse(BaseModel):
    statement: str
    cited_verse_ids: list[str]


class AnswerResponse(BaseModel):
    session_id: str
    query: str
    retrieval_query: str
    response_language: SupportedLanguage
    mode: AnswerMode
    evidence_sufficient: bool
    scriptural_teaching: list[ScripturalTeachingResponse]
    interpretation: str | None
    application: str | None
    limitations: list[str]
    cited_verse_ids: list[str]
    retrieved_verse_ids: list[str]


def create_vedavault_service() -> VedaVaultService:
    """Construct the frozen V1 stack from local resources and environment configuration."""
    configured_path = os.environ.get("VEDAVAULT_INDEX_PATH")
    index_path = (
        Path(configured_path).expanduser()
        if configured_path
        else DEFAULT_INDEX_PATH
    )
    embedding_provider = SentenceTransformerEmbeddingProvider(local_files_only=True)
    vector_store = LocalVectorStore.load(
        index_path,
        embedding_provider=embedding_provider,
    )
    groq_client = GroqClient()
    return VedaVaultService(
        GroqQueryUnderstandingProvider(client=groq_client),
        Retriever(embedding_provider, vector_store),
        GroqLLMProvider(client=groq_client),
    )


@lru_cache(maxsize=1)
def get_vedavault_service() -> VedaVaultService:
    """Lazily create one reusable service instance for this API process."""
    return create_vedavault_service()


@lru_cache(maxsize=1)
def get_conversation_store() -> InMemoryConversationStore:
    """Return one process-local, thread-safe conversation store."""
    return InMemoryConversationStore()


def _cors_origins() -> tuple[str, ...]:
    configured = os.environ.get("VEDAVAULT_CORS_ORIGINS")
    if configured is None:
        return DEFAULT_CORS_ORIGINS
    origins = tuple(origin.strip() for origin in configured.split(",") if origin.strip())
    if not origins:
        raise ValueError("VEDAVAULT_CORS_ORIGINS must contain at least one origin")
    if "*" in origins:
        raise ValueError("VEDAVAULT_CORS_ORIGINS must not contain an unrestricted origin")
    return origins


def _error_response(status_code: int, error: str, message: str, **details: object) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "message": message, **details},
    )


def _is_rate_limit_error(error: Exception) -> bool:
    normalized = " ".join(str(error).casefold().split())
    return "http 429" in normalized or "rate limit" in normalized or "rate_limit" in normalized


def _serialize_response(response: VedaVaultResponse, session_id: str) -> AnswerResponse:
    answer = response.answer
    return AnswerResponse(
        session_id=session_id,
        query=response.original_query,
        retrieval_query=response.retrieval_query,
        response_language=response.language_policy.effective_primary_response_language,
        mode=answer.mode,
        evidence_sufficient=answer.evidence_sufficient,
        scriptural_teaching=[
            ScripturalTeachingResponse(
                statement=claim.statement,
                cited_verse_ids=list(claim.cited_verse_ids),
            )
            for claim in answer.scriptural_claims
        ],
        interpretation=answer.interpretation,
        application=answer.application,
        limitations=list(answer.limitations),
        cited_verse_ids=list(answer.cited_verse_ids),
        retrieved_verse_ids=list(response.retrieved_passage_ids),
    )


def _assistant_history_text(response: VedaVaultResponse) -> str:
    answer = response.answer
    parts = [claim.statement for claim in answer.scriptural_claims]
    parts.extend(
        value
        for value in (answer.interpretation, answer.application)
        if value is not None
    )
    parts.extend(f"Limitation: {value}" for value in answer.limitations)
    return "\n".join(parts) or "No answer text was returned."


def create_app() -> FastAPI:
    api = FastAPI(title="VedaVault API", version="1.0.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=list(_cors_origins()),
        allow_credentials=False,
        allow_methods=("GET", "POST", "DELETE"),
        allow_headers=("Content-Type",),
    )

    @api.exception_handler(ClarificationRequiredError)
    async def clarification_required_handler(
        _request: Request, error: ClarificationRequiredError
    ) -> JSONResponse:
        return _error_response(
            409,
            "clarification_required",
            "The query needs clarification before scripture can be retrieved.",
            query=error.understanding.original_query,
            clarification_required=True,
        )

    @api.exception_handler(EvidenceHygieneError)
    async def evidence_hygiene_handler(
        _request: Request, _error: EvidenceHygieneError
    ) -> JSONResponse:
        return _error_response(
            503,
            "evidence_unavailable",
            "Safe scriptural evidence is temporarily unavailable.",
        )

    @api.exception_handler(LLMProviderError)
    @api.exception_handler(QueryUnderstandingProviderError)
    async def provider_error_handler(
        _request: Request, error: Exception
    ) -> JSONResponse:
        if _is_rate_limit_error(error):
            return _error_response(
                429,
                "upstream_rate_limited",
                "The upstream generation service is rate limited. Try again later.",
            )
        return _error_response(
            502,
            "upstream_service_error",
            "The upstream generation service could not complete the request.",
        )

    @api.exception_handler(IndexCompatibilityError)
    @api.exception_handler(IndexManifestError)
    @api.exception_handler(RuntimeError)
    async def service_error_handler(
        _request: Request, _error: Exception
    ) -> JSONResponse:
        return _error_response(
            503,
            "service_unavailable",
            "VedaVault is temporarily unavailable.",
        )

    @api.exception_handler(Exception)
    async def unexpected_error_handler(
        _request: Request, _error: Exception
    ) -> JSONResponse:
        return _error_response(
            500,
            "internal_error",
            "An unexpected server error occurred.",
        )

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "vedavault", "rag_version": "v1"}

    @api.post("/answer", response_model=AnswerResponse)
    def answer(
        payload: AnswerRequest,
        service: VedaVaultService = Depends(get_vedavault_service),
        conversation_store: ConversationStore = Depends(get_conversation_store),
    ) -> AnswerResponse:
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="query must be a non-empty string")
        if (
            payload.input_language is not None
            and payload.input_language not in V1_LANGUAGES
        ):
            raise HTTPException(status_code=400, detail="unsupported V1 input language")
        if (
            payload.response_language is not None
            and payload.response_language not in V1_LANGUAGES
        ):
            raise HTTPException(status_code=400, detail="unsupported V1 response language")
        if payload.session_id is not None and not payload.session_id.strip():
            raise HTTPException(status_code=400, detail="session_id must be non-empty")

        if payload.session_id is None:
            conversation_context = ConversationContext()
        else:
            session = conversation_store.get_session(payload.session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="unknown session_id")
            conversation_context = session.context

        language_policy = LanguagePolicy(
            input_languages=(
                (payload.input_language,)
                if payload.input_language is not None
                else ()
            ),
            conversation_language=conversation_context.latest_response_language,
            requested_response_language=payload.response_language,
        )
        response = service.answer(
            payload.query,
            language_policy,
            mode=payload.mode,
            conversation_context=conversation_context,
        )
        session_id = payload.session_id
        if session_id is None:
            session_id = conversation_store.create_session().session_id
        try:
            conversation_store.append_exchange(
                session_id,
                payload.query,
                _assistant_history_text(response),
                response.language_policy.effective_primary_response_language,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown session_id") from exc
        return _serialize_response(response, session_id)

    @api.delete("/sessions/{session_id}")
    def delete_session(
        session_id: str,
        conversation_store: ConversationStore = Depends(get_conversation_store),
    ) -> dict[str, str]:
        if not conversation_store.delete_session(session_id):
            raise HTTPException(status_code=404, detail="unknown session_id")
        return {"status": "deleted", "session_id": session_id}

    return api


app = create_app()
