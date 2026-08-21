"""Groq-backed query understanding and grounded answer generation."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .answer import AnswerContract, AnswerMode, ScripturalClaim
from .conversation import ConversationContext
from .language import LanguagePolicy, SupportedLanguage
from .llm import GenerationRequest, LLMProvider, LLMProviderError
from .query_understanding import (
    QUERY_UNDERSTANDING_INSTRUCTIONS,
    QueryUnderstandingProvider,
    QueryUnderstandingProviderError,
    QueryUnderstandingResult,
    minimal_contextual_retrieval_query,
)


DEFAULT_GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "qwen/qwen3.6-27b"
DEFAULT_GROQ_TIMEOUT_SECONDS = 30.0
GROQ_REASONING_EFFORT = "none"
GROQ_TEMPERATURE = 0.0
GROQ_USER_AGENT = "VedaVault/1.0 Python"


GROUNDED_GENERATION_INSTRUCTIONS = """GROUNDED GENERATION REQUIREMENTS
- Use only the supplied scriptural evidence. SCRIPTURE CLAIMS ARE EVIDENCE-ONLY.
  Every scriptural_teaching statement must
  be a faithful paraphrase of the text in its cited supplied evidence.
- Never introduce Bhagavad Gita doctrine from pretrained memory, even when it is
  correct, famous, or highly relevant to the user's question. Never mention or
  paraphrase a teaching associated with a verse that was not supplied.
- If you know a relevant Gita teaching but it is absent from the supplied
  evidence, do not use it. General model knowledge must never fill an evidence gap.
- Interpretation and application may explain, contextualize, or give practical
  advice derived from supplied teachings. They must not smuggle in a new
  scriptural proposition unsupported by the supplied evidence. A supported
  implication is allowed; a new doctrine from model memory is forbidden.
- Treat retrieved passages as candidate evidence, not mandatory citations.
- Irrelevant or weak passages may be ignored and must not be forced into the
  answer. Cite only passages that materially support the final answer and prefer
  the smallest sufficient citation set and evidence subset. Weaker or secondary
  passages may be omitted. Citing one supplied verse out of five is acceptable.
  Never cite a passage merely because retrieval returned it.
- Never fabricate scripture, quotations, or canonical verse identifiers.
- Preserve supplied Sanskrit exactly whenever quoting it.
- Never fabricate a translation. Never represent generated explanation or
  translation as canonical scripture. Never represent interpretation,
  application, or commentary as direct canonical scripture.
- Keep scriptural teaching, interpretation, and application in separate fields.
- Answer primarily in the resolved response language; an explicit requested
  response language overrides automatic language choice.
- Cite only canonical passage IDs present in the supplied evidence.
- When evidence is sufficient, every scriptural teaching must have at least one
  supplied citation whose text directly supports that statement.
- If evidence only partially addresses the question, or the supplied evidence
  otherwise does not fully answer it, set
  evidence_sufficient=false, include a concise non-empty limitation, and answer
  only the portion directly supported. Do not fill gaps from model memory or
  otherwise compensate with model knowledge. A shorter incomplete-but-grounded answer is preferable to a
  complete-looking unsupported answer.
- Return only one JSON object matching the requested answer schema."""


RESPONSE_LANGUAGE_INSTRUCTIONS = {
    SupportedLanguage.ENGLISH: (
        "Write generated prose in natural English."
    ),
    SupportedLanguage.HINDI: (
        "Write generated prose in natural Hindi using Devanagari script. "
        "Romanized Hindi or Hinglish input does not authorize Romanized output."
    ),
    SupportedLanguage.BENGALI: (
        "Write generated prose in natural Bengali using Bengali script. "
        "Romanized Bengali or Banglish input does not authorize Romanized output."
    ),
    SupportedLanguage.SANSKRIT: (
        "Write generated prose in Sanskrit using Devanagari script when Sanskrit "
        "is the resolved response language. Never fabricate Sanskrit text."
    ),
}


class GroqTransport(Protocol):
    """Minimal injectable HTTP boundary used by offline tests and the real client."""

    def post(
        self,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> bytes:
        """POST JSON bytes and return response bytes."""


class _GroqClientError(RuntimeError):
    pass


class _UrllibTransport:
    def post(
        self,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> bytes:
        request_headers = dict(headers)
        request_headers["Content-Type"] = "application/json"
        request_headers["Accept"] = "application/json"
        request_headers["User-Agent"] = GROQ_USER_AGENT
        request = urllib.request.Request(
            url,
            data=body,
            headers=request_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return response.read()
        except urllib.error.HTTPError as exc:
            raise _safe_http_error(exc, request_headers) from exc


class GroqClient:
    """Small OpenAI-compatible Groq chat client with deterministic defaults."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = DEFAULT_GROQ_ENDPOINT,
        model: str = DEFAULT_GROQ_MODEL,
        timeout_seconds: float = DEFAULT_GROQ_TIMEOUT_SECONDS,
        transport: GroqTransport | None = None,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(timeout_seconds, int | float) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self._transport = transport or _UrllibTransport()

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_completion_tokens: int,
    ) -> str:
        """Return one JSON-mode assistant message or raise an internal clean error."""
        if not isinstance(self._api_key, str) or not self._api_key.strip():
            raise _GroqClientError("GROQ_API_KEY is not configured")
        if not isinstance(max_completion_tokens, int) or isinstance(max_completion_tokens, bool) or max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be a positive integer")
        normalized_messages = [dict(message) for message in messages]
        if not normalized_messages or any(
            set(message) != {"role", "content"}
            or message["role"] not in {"system", "user", "assistant"}
            or not isinstance(message["content"], str)
            or not message["content"].strip()
            for message in normalized_messages
        ):
            raise ValueError("messages must contain role/content string mappings")
        payload = {
            "model": self.model,
            "messages": normalized_messages,
            "temperature": GROQ_TEMPERATURE,
            "reasoning_effort": GROQ_REASONING_EFFORT,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_completion_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": GROQ_USER_AGENT,
        }
        try:
            response_bytes = self._transport.post(
                self.endpoint,
                headers,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                self.timeout_seconds,
            )
        except _GroqClientError:
            raise
        except Exception as exc:
            raise _GroqClientError("Groq request failed") from exc
        try:
            response = json.loads(response_bytes.decode("utf-8"))
            choices = response["choices"]
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _GroqClientError("Groq returned a malformed API response") from exc
        if not isinstance(content, str) or not content.strip():
            raise _GroqClientError("Groq returned an empty model response")
        return content


class GroqQueryUnderstandingProvider(QueryUnderstandingProvider):
    """Groq implementation of retrieval rewrite; it never creates evidence."""

    def __init__(self, client: GroqClient | None = None, **client_options: Any) -> None:
        if client is not None and client_options:
            raise ValueError("client cannot be combined with client options")
        self.client = client or GroqClient(**client_options)

    def understand(
        self,
        original_query: str,
        language_policy: LanguagePolicy,
        conversation_context: ConversationContext | None = None,
    ) -> QueryUnderstandingResult:
        if not isinstance(original_query, str) or not original_query.strip():
            raise ValueError("original_query must be a non-empty string")
        if not isinstance(language_policy, LanguagePolicy):
            raise ValueError("language_policy must be a LanguagePolicy")
        if conversation_context is not None and not isinstance(
            conversation_context, ConversationContext
        ):
            raise ValueError("conversation_context must be a ConversationContext or None")
        query_input = {
            "original_query": original_query,
            "language_policy": language_policy.to_dict(),
            "required_output": {
                "retrieval_query": "concise semantic retrieval intent",
                "clarification_required": False,
            },
        }
        if conversation_context is not None:
            query_input["conversation_context"] = conversation_context.to_dict()
        user_input = json.dumps(
            query_input,
            ensure_ascii=False,
            sort_keys=True,
        )
        try:
            content = self.client.complete(
                (
                    {"role": "system", "content": QUERY_UNDERSTANDING_INSTRUCTIONS},
                    {"role": "user", "content": user_input},
                ),
                max_completion_tokens=256,
            )
            value = _model_json_object(content, "query-understanding")
            if set(value) != {"retrieval_query", "clarification_required"}:
                raise ValueError("query-understanding output has unexpected fields")
            retrieval_query = value["retrieval_query"]
            clarification_required = value["clarification_required"]
            if not isinstance(retrieval_query, str) or not retrieval_query.strip():
                raise ValueError("retrieval_query must be a non-empty string")
            if conversation_context is not None:
                retrieval_query = minimal_contextual_retrieval_query(retrieval_query)
            if not isinstance(clarification_required, bool):
                raise ValueError("clarification_required must be boolean")
            result = QueryUnderstandingResult(
                original_query=original_query,
                retrieval_query=retrieval_query,
                language_policy=language_policy,
                clarification_required=clarification_required or language_policy.clarification_needed,
            )
        except _GroqClientError as exc:
            raise QueryUnderstandingProviderError(str(exc)) from exc
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise QueryUnderstandingProviderError("Groq returned invalid query-understanding JSON") from exc
        return self.validate_response(original_query, language_policy, result)


class GroqLLMProvider(LLMProvider):
    """Groq grounded-generation provider returning the existing AnswerContract."""

    def __init__(self, client: GroqClient | None = None, **client_options: Any) -> None:
        if client is not None and client_options:
            raise ValueError("client cannot be combined with client options")
        self.client = client or GroqClient(**client_options)

    def generate(self, request: GenerationRequest) -> AnswerContract:
        if not isinstance(request, GenerationRequest):
            raise ValueError("request must be a GenerationRequest")
        try:
            content = self.client.complete(
                (
                    {"role": "system", "content": _generation_system_prompt(request)},
                    {"role": "user", "content": request.grounding_context.to_prompt_context()},
                ),
                max_completion_tokens=1400,
            )
            answer = _answer_from_json(_model_json_object(content, "answer"))
        except _GroqClientError as exc:
            raise LLMProviderError(str(exc)) from exc
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LLMProviderError("Groq returned invalid answer JSON") from exc
        return self.validate_response(request, answer)


def _generation_system_prompt(request: GenerationRequest) -> str:
    policy = request.language_policy or LanguagePolicy()
    supplied_ids = list(
        dict.fromkeys(
            item.passage_id
            for item in request.grounding_context.evidence_items
            if item.passage_id is not None
        )
    )
    schema = {
        "query": request.query,
        "mode": request.mode.value,
        "scriptural_teaching": [
            {"statement": "supported teaching", "cited_verse_ids": ["supplied canonical passage ID"]}
        ],
        "interpretation": (
            "required non-empty interpretation when evidence_sufficient is true"
            if request.mode is AnswerMode.PHILOSOPHICAL
            else None
        ),
        "application": (
            "required non-empty application when evidence_sufficient is true"
            if request.mode is AnswerMode.APPLICATION
            else None
        ),
        "evidence_sufficient": True,
        "limitations": [],
        "evidence_passage_ids": supplied_ids,
    }
    return "\n\n".join(
        (
            GROUNDED_GENERATION_INSTRUCTIONS,
            request.grounding_rules,
            request.answer_contract_rules,
            "LANGUAGE POLICY\n" + json.dumps(policy.to_dict(), ensure_ascii=False, sort_keys=True),
            _response_language_prompt(policy),
            f"ANSWER MODE\n{request.mode.value}",
            _answer_mode_quality_prompt(request.mode),
            "SUPPLIED CANONICAL PASSAGE IDS\n" + json.dumps(supplied_ids, ensure_ascii=False),
            "REQUIRED JSON SHAPE\n" + json.dumps(schema, ensure_ascii=False, sort_keys=True),
            "The query and mode must exactly match the required JSON shape. "
            "evidence_passage_ids must contain exactly the supplied canonical passage IDs. "
            "That traceability field does not mean every supplied passage must be cited. "
            "Use cited_verse_ids only for the smallest set that materially supports each claim. "
            "Use null for semantic layers excluded by the requested answer mode.",
        )
    )


def _response_language_prompt(policy: LanguagePolicy) -> str:
    language = policy.effective_primary_response_language
    instruction = RESPONSE_LANGUAGE_INSTRUCTIONS.get(
        language,
        "Write generated prose in the resolved response language.",
    )
    return (
        f"RESOLVED RESPONSE LANGUAGE\n{language.value}\n{instruction}\n"
        "The resolved response language already applies any explicit requested "
        "response-language override and is authoritative. Input script, "
        "transliteration, Hinglish, Banglish, or code-switching must not override it."
    )


def _answer_mode_quality_prompt(mode: AnswerMode) -> str:
    if mode is AnswerMode.TEXTUAL:
        return """TEXTUAL MODE REQUIREMENTS
Be concise, synthesize only the strongest relevant supplied evidence, and avoid
repetitive verse-by-verse paraphrases or unnecessary padding.
Set interpretation=null and application=null."""
    if mode is AnswerMode.PHILOSOPHICAL:
        return """PHILOSOPHICAL MODE REQUIREMENTS
Be concise and derive the philosophical explanation only from supported supplied
teachings. If evidence_sufficient=true, interpretation must be a non-null,
non-empty string. Set application=null. If evidence is insufficient, report the
limitation instead of introducing doctrine from memory."""
    return """APPLICATION MODE QUALITY
1. Give a concise scriptural teaching grounded in the strongest supplied evidence.
2. Explain clearly what that teaching means without presenting interpretation as scripture.
3. Give a practical application addressing the user's actual situation.
If evidence_sufficient=true, application must be a non-null, non-empty string and
must be derived from the supplied teaching. Do not add unsupported Gita doctrine
to make the advice seem more complete.
Avoid repetitive verse-by-verse paraphrases, unnecessary philosophical padding,
and using every retrieved passage merely because it was supplied."""


def _safe_http_error(
    error: urllib.error.HTTPError,
    request_headers: Mapping[str, str],
) -> _GroqClientError:
    parts = [f"Groq HTTP {error.code}"]
    try:
        body = error.read()
        payload = json.loads(body.decode("utf-8"))
    except (AttributeError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        details = payload["error"]
        for field_name in ("code", "type", "message"):
            value = details.get(field_name)
            if isinstance(value, str | int | float) and not isinstance(value, bool):
                normalized = " ".join(str(value).split())
                if normalized:
                    parts.append(f"{field_name}={normalized[:500]}")
    return _GroqClientError(_redact_request_secrets("; ".join(parts), request_headers))


def _redact_request_secrets(
    diagnostic: str,
    request_headers: Mapping[str, str],
) -> str:
    authorization = next(
        (
            value
            for name, value in request_headers.items()
            if name.lower() == "authorization"
        ),
        "",
    )
    secrets = [authorization]
    if authorization.lower().startswith("bearer "):
        secrets.append(authorization[7:].strip())
    for secret in sorted(
        (value for value in secrets if value),
        key=len,
        reverse=True,
    ):
        diagnostic = diagnostic.replace(secret, "[REDACTED]")
    return diagnostic


def _model_json_object(content: str, label: str) -> dict[str, Any]:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError(f"{label} output must be a JSON object")
    return value


def _answer_from_json(value: Mapping[str, Any]) -> AnswerContract:
    required_fields = {
        "query",
        "mode",
        "scriptural_teaching",
        "interpretation",
        "application",
        "evidence_sufficient",
        "limitations",
        "evidence_passage_ids",
    }
    if set(value) != required_fields:
        raise ValueError("answer output has unexpected fields")
    if not isinstance(value["query"], str):
        raise ValueError("answer query must be a string")
    mode = AnswerMode(value["mode"])
    teaching = value["scriptural_teaching"]
    if not isinstance(teaching, list):
        raise ValueError("scriptural_teaching must be a list")
    claims = []
    for claim in teaching:
        if not isinstance(claim, dict) or set(claim) != {"statement", "cited_verse_ids"}:
            raise ValueError("scriptural teaching entries have invalid fields")
        citations = claim["cited_verse_ids"]
        if not isinstance(citations, list):
            raise ValueError("cited_verse_ids must be a list")
        claims.append(ScripturalClaim(claim["statement"], tuple(citations)))
    interpretation = _optional_non_empty_string(value["interpretation"], "interpretation")
    application = _optional_non_empty_string(value["application"], "application")
    evidence_sufficient = value["evidence_sufficient"]
    if not isinstance(evidence_sufficient, bool):
        raise ValueError("evidence_sufficient must be boolean")
    limitations = value["limitations"]
    evidence_ids = value["evidence_passage_ids"]
    if not isinstance(limitations, list) or not isinstance(evidence_ids, list):
        raise ValueError("limitations and evidence_passage_ids must be lists")
    return AnswerContract(
        query=value["query"],
        mode=mode,
        scriptural_claims=tuple(claims),
        interpretation=interpretation,
        application=application,
        evidence_sufficient=evidence_sufficient,
        limitations=tuple(limitations),
        evidence_passage_ids=frozenset(evidence_ids),
    )


def _optional_non_empty_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string or null")
    return value
