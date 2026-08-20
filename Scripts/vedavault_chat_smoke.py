"""Manual one-query VedaVault smoke test. Never invoked by automated tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    AnswerMode,
    ClarificationRequiredError,
    GroqClient,
    GroqLLMProvider,
    GroqQueryUnderstandingProvider,
    LanguagePolicy,
    LocalVectorStore,
    Retriever,
    SentenceTransformerEmbeddingProvider,
    SupportedLanguage,
    VedaVaultService,
)


INDEX_DIRECTORY = (
    ROOT / "Data" / "Processed" / "Bhagavad_Gita" / "retrieval_index"
)
V1_LANGUAGES = {
    language.value: language
    for language in (
        SupportedLanguage.ENGLISH,
        SupportedLanguage.HINDI,
        SupportedLanguage.BENGALI,
        SupportedLanguage.SANSKRIT,
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one explicitly approved live VedaVault query."
    )
    parser.add_argument("query", help="Original user query")
    parser.add_argument(
        "--input-language",
        choices=tuple(V1_LANGUAGES),
        default="en",
        help="Current-turn language policy hint (default: en)",
    )
    parser.add_argument(
        "--response-language",
        choices=tuple(V1_LANGUAGES),
        help="Explicit response-language override",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in AnswerMode),
        default=AnswerMode.TEXTUAL.value,
        help="Answer contract mode (default: textual)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not INDEX_DIRECTORY.is_dir():
        raise SystemExit(f"Existing retrieval index not found: {INDEX_DIRECTORY}")

    language_policy = LanguagePolicy(
        input_languages=(V1_LANGUAGES[args.input_language],),
        requested_response_language=(
            V1_LANGUAGES[args.response_language]
            if args.response_language is not None
            else None
        ),
    )
    embedding_provider = SentenceTransformerEmbeddingProvider(local_files_only=True)
    vector_store = LocalVectorStore.load(
        INDEX_DIRECTORY,
        embedding_provider=embedding_provider,
    )
    groq_client = GroqClient()
    service = VedaVaultService(
        GroqQueryUnderstandingProvider(client=groq_client),
        Retriever(embedding_provider, vector_store),
        GroqLLMProvider(client=groq_client),
    )

    try:
        response = service.answer(
            args.query,
            language_policy,
            mode=AnswerMode(args.mode),
        )
    except ClarificationRequiredError as exc:
        print(f"Original query: {exc.understanding.original_query}")
        print(f"Retrieval rewrite: {exc.understanding.retrieval_query}")
        print("Clarification required; retrieval and generation were skipped.")
        return 2

    print(f"Original query: {response.original_query}")
    print(f"Retrieval rewrite: {response.retrieval_query}")
    print("Retrieved canonical verse IDs:")
    for passage_id in response.retrieved_passage_ids:
        print(f"- {passage_id}")
    print("AnswerContract:")
    print(
        json.dumps(
            json.loads(response.answer.to_json()),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
