"""Bounded process-local conversation context; never a scripture evidence source."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from uuid import uuid4

from .language import SupportedLanguage


V1_CONVERSATION_TURN_LIMIT = 8
V1_USER_HISTORY_CHAR_LIMIT = 2000
V1_ASSISTANT_HISTORY_CHAR_LIMIT = 2000


class ConversationRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ConversationTurn:
    """One compact text-only turn retained for contextual query understanding."""

    role: ConversationRole
    text: str
    response_language: SupportedLanguage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, ConversationRole):
            raise ValueError("role must be a ConversationRole")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("conversation turn text must be a non-empty string")
        if self.response_language is not None and not isinstance(
            self.response_language, SupportedLanguage
        ):
            raise ValueError("response_language must be a SupportedLanguage or None")
        if self.role is ConversationRole.USER and self.response_language is not None:
            raise ValueError("user turns must not declare a response language")
        if self.role is ConversationRole.ASSISTANT and self.response_language is None:
            raise ValueError("assistant turns require their resolved response language")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "role": self.role.value,
            "text": self.text,
            "response_language": (
                self.response_language.value
                if self.response_language is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ConversationContext:
    """Immutable bounded history supplied only to query understanding."""

    turns: tuple[ConversationTurn, ...] = ()

    def __post_init__(self) -> None:
        turns = tuple(self.turns)
        if not all(isinstance(turn, ConversationTurn) for turn in turns):
            raise ValueError("turns must contain ConversationTurn values")
        if len(turns) > V1_CONVERSATION_TURN_LIMIT:
            raise ValueError("conversation context exceeds the V1 turn limit")
        for turn in turns:
            limit = (
                V1_USER_HISTORY_CHAR_LIMIT
                if turn.role is ConversationRole.USER
                else V1_ASSISTANT_HISTORY_CHAR_LIMIT
            )
            if len(turn.text) > limit:
                raise ValueError("conversation turn text exceeds the V1 character limit")
        object.__setattr__(self, "turns", turns)

    @property
    def latest_response_language(self) -> SupportedLanguage | None:
        for turn in reversed(self.turns):
            if turn.role is ConversationRole.ASSISTANT:
                return turn.response_language
        return None

    def to_dict(self) -> dict[str, list[dict[str, str | None]]]:
        return {"recent_turns": [turn.to_dict() for turn in self.turns]}


@dataclass(frozen=True)
class ConversationSession:
    """Immutable snapshot returned by a conversation store."""

    session_id: str
    turns: tuple[ConversationTurn, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        turns = tuple(self.turns)
        if not all(isinstance(turn, ConversationTurn) for turn in turns):
            raise ValueError("turns must contain ConversationTurn values")
        ConversationContext(turns)
        object.__setattr__(self, "turns", turns)

    @property
    def context(self) -> ConversationContext:
        return ConversationContext(self.turns)


class ConversationStore(ABC):
    """Provider-neutral process-local session storage boundary."""

    @abstractmethod
    def create_session(self) -> ConversationSession:
        """Create and return an empty session with an opaque identifier."""

    @abstractmethod
    def get_session(self, session_id: str) -> ConversationSession | None:
        """Return an immutable snapshot, or None when the ID is unknown."""

    @abstractmethod
    def append_exchange(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        response_language: SupportedLanguage,
    ) -> ConversationSession:
        """Atomically append one successful user/assistant exchange."""

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and report whether it existed."""


class InMemoryConversationStore(ConversationStore):
    """Thread-safe bounded session snapshots for one API process."""

    def __init__(self, *, max_turns: int = V1_CONVERSATION_TURN_LIMIT) -> None:
        if (
            not isinstance(max_turns, int)
            or isinstance(max_turns, bool)
            or max_turns < 2
            or max_turns > V1_CONVERSATION_TURN_LIMIT
            or max_turns % 2 != 0
        ):
            raise ValueError(
                "max_turns must be an even integer from 2 through the V1 turn limit"
            )
        self.max_turns = max_turns
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = RLock()

    def create_session(self) -> ConversationSession:
        with self._lock:
            session_id = str(uuid4())
            while session_id in self._sessions:
                session_id = str(uuid4())
            session = ConversationSession(session_id)
            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> ConversationSession | None:
        if not isinstance(session_id, str) or not session_id.strip():
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def append_exchange(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        response_language: SupportedLanguage,
    ) -> ConversationSession:
        user_turn = ConversationTurn(
            ConversationRole.USER,
            _bounded_text(user_text, V1_USER_HISTORY_CHAR_LIMIT),
        )
        assistant_turn = ConversationTurn(
            ConversationRole.ASSISTANT,
            _bounded_text(assistant_text, V1_ASSISTANT_HISTORY_CHAR_LIMIT),
            response_language,
        )
        with self._lock:
            current = self._sessions.get(session_id)
            if current is None:
                raise KeyError(session_id)
            turns = (current.turns + (user_turn, assistant_turn))[-self.max_turns :]
            updated = ConversationSession(session_id, turns)
            self._sessions[session_id] = updated
            return updated

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)


def _bounded_text(text: str, limit: int) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("conversation text must be a non-empty string")
    return text[:limit]
