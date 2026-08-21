"""Offline tests for bounded process-local conversation state."""

from __future__ import annotations

import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from uuid import UUID


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    ConversationContext,
    ConversationRole,
    ConversationTurn,
    InMemoryConversationStore,
    SupportedLanguage,
    V1_CONVERSATION_TURN_LIMIT,
)


class ConversationStoreTests(unittest.TestCase):
    def test_new_session_uses_an_opaque_uuid_and_immutable_snapshot(self) -> None:
        store = InMemoryConversationStore()
        session = store.create_session()
        self.assertEqual(str(UUID(session.session_id)), session.session_id)
        self.assertEqual(session.turns, ())
        with self.assertRaises(FrozenInstanceError):
            session.session_id = "changed"  # type: ignore[misc]

    def test_exchange_order_is_preserved_and_snapshots_are_defensive(self) -> None:
        store = InMemoryConversationStore()
        initial = store.create_session()
        updated = store.append_exchange(
            initial.session_id,
            "first user turn",
            "first assistant turn",
            SupportedLanguage.BENGALI,
        )
        self.assertEqual(initial.turns, ())
        self.assertEqual(
            [(turn.role, turn.text) for turn in updated.turns],
            [
                (ConversationRole.USER, "first user turn"),
                (ConversationRole.ASSISTANT, "first assistant turn"),
            ],
        )
        self.assertEqual(
            updated.context.latest_response_language,
            SupportedLanguage.BENGALI,
        )

    def test_history_is_bounded_and_discards_oldest_exchanges(self) -> None:
        store = InMemoryConversationStore(max_turns=V1_CONVERSATION_TURN_LIMIT)
        session_id = store.create_session().session_id
        for index in range(6):
            store.append_exchange(
                session_id,
                f"user-{index}",
                f"assistant-{index}",
                SupportedLanguage.ENGLISH,
            )
        turns = store.get_session(session_id).turns
        self.assertEqual(len(turns), V1_CONVERSATION_TURN_LIMIT)
        self.assertEqual(turns[0].text, "user-2")
        self.assertEqual(turns[-1].text, "assistant-5")

    def test_delete_and_unknown_session_policy_are_deterministic(self) -> None:
        store = InMemoryConversationStore()
        session_id = store.create_session().session_id
        self.assertTrue(store.delete_session(session_id))
        self.assertFalse(store.delete_session(session_id))
        self.assertIsNone(store.get_session(session_id))
        with self.assertRaises(KeyError):
            store.append_exchange(
                session_id,
                "user",
                "assistant",
                SupportedLanguage.ENGLISH,
            )

    def test_concurrent_exchange_appends_never_interleave_turn_pairs(self) -> None:
        store = InMemoryConversationStore()
        session_id = store.create_session().session_id

        def append(index: int) -> None:
            store.append_exchange(
                session_id,
                f"user-{index}",
                f"assistant-{index}",
                SupportedLanguage.ENGLISH,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(append, range(20)))

        turns = store.get_session(session_id).turns
        self.assertEqual(len(turns), V1_CONVERSATION_TURN_LIMIT)
        for index in range(0, len(turns), 2):
            user_turn, assistant_turn = turns[index : index + 2]
            self.assertIs(user_turn.role, ConversationRole.USER)
            self.assertIs(assistant_turn.role, ConversationRole.ASSISTANT)
            self.assertEqual(
                assistant_turn.text.removeprefix("assistant-"),
                user_turn.text.removeprefix("user-"),
            )

    def test_context_serializes_text_and_language_without_evidence_fields(self) -> None:
        context = ConversationContext(
            (
                ConversationTurn(ConversationRole.USER, "and desire?"),
                ConversationTurn(
                    ConversationRole.ASSISTANT,
                    "A compact prior answer.",
                    SupportedLanguage.ENGLISH,
                ),
            )
        )
        serialized = context.to_dict()
        self.assertEqual(
            [turn["role"] for turn in serialized["recent_turns"]],
            ["user", "assistant"],
        )
        self.assertNotIn("evidence", repr(serialized).casefold())
        self.assertNotIn("citation", repr(serialized).casefold())

    def test_context_and_store_reject_configuration_above_v1_bounds(self) -> None:
        turns = tuple(
            ConversationTurn(ConversationRole.USER, f"turn-{index}")
            for index in range(V1_CONVERSATION_TURN_LIMIT + 1)
        )
        with self.assertRaisesRegex(ValueError, "V1 turn limit"):
            ConversationContext(turns)
        with self.assertRaisesRegex(ValueError, "V1 turn limit"):
            InMemoryConversationStore(max_turns=V1_CONVERSATION_TURN_LIMIT + 2)


if __name__ == "__main__":
    unittest.main()
