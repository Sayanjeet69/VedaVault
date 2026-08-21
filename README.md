# ॐ VedaVault

## Multilingual, Citation-Grounded AI for Indian Scripture

VedaVault is a multilingual Retrieval-Augmented Generation (RAG) application for exploring Indian scripture through natural-language conversation.

The current V1 focuses on the **Bhagavad Gita** and supports English, Hindi, Bengali, Sanskrit, transliterated Sanskrit, Hinglish, Banglish, and code-switched queries.

Unlike a normal LLM chatbot, VedaVault retrieves relevant scripture first and generates answers from supplied evidence with verse-level citations.

> **Ancient wisdom. Grounded answers. Modern clarity.**

---

## Current V1 Capabilities

- Bhagavad Gita question answering
- English, Hindi, Bengali and Sanskrit
- Hinglish and Banglish
- Romanized Sanskrit / transliteration
- Code-switched queries
- Imperfect spelling and grammar tolerance
- Cross-language semantic retrieval
- Textual, Philosophical and Application answer modes
- Explicit response-language selection
- Multi-turn conversations
- Session-based contextual follow-ups
- Fresh retrieval on every turn
- Verse-level citations
- Evidence sufficiency handling
- Responsive desktop and mobile UI
- New Journey / session reset

---

## Architecture

```text
Angular Frontend
      │
      │ REST / JSON
      ▼
FastAPI
      │
Session Context
      ▼
Language Policy
      ▼
Query Understanding Provider
      ▼
Retriever
      ▼
EvidenceBundle
      ▼
Evidence Hygiene
      ▼
GroundingContext
      ▼
GenerationRequest
      ▼
LLMProvider — Qwen via Groq
      ▼
AnswerContract
      ▼
Validated Grounded Response

The architecture is modular so retrieval, grounding, generation, language handling, and frontend/backend integration can evolve independently.

VedaVault/
│
├── Backend/
│   ├── vedavault_api.py
│   ├── requirements-api.txt
│   ├── README.md
│   │
│   └── vedavault_retrieval/
│       ├── __init__.py
│       ├── application.py
│       ├── conversation.py
│       ├── evidence_hygiene.py
│       ├── groq.py
│       ├── language.py
│       ├── llm.py
│       ├── query_understanding.py
│       ├── retrieval-related modules
│       ├── evidence / grounding / answer-contract modules
│       └── ...
│
├── Frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── core/
│   │   │   │   ├── config/
│   │   │   │   ├── models/
│   │   │   │   └── services/
│   │   │   │
│   │   │   ├── components/
│   │   │   │   ├── assistant-message/
│   │   │   │   ├── brand-mark/
│   │   │   │   ├── chat-composer/
│   │   │   │   ├── citation-chip/
│   │   │   │   ├── error-notice/
│   │   │   │   ├── scripture-card/
│   │   │   │   ├── sidebar/
│   │   │   │   ├── thinking-indicator/
│   │   │   │   └── user-message/
│   │   │   │
│   │   │   ├── pages/
│   │   │   │   ├── welcome/
│   │   │   │   └── chat/
│   │   │   │
│   │   │   ├── app.config.ts
│   │   │   ├── app.routes.ts
│   │   │   └── ...
│   │   │
│   │   ├── styles.css
│   │   └── index.html
│   │
│   ├── angular.json
│   ├── package.json
│   ├── package-lock.json
│   └── .postcssrc.json
│
├── Data/
│   ├── Raw/
│   └── Processed/
│       └── Bhagavad_Gita/
│           ├── corpus.json
│           └── retrieval_index/
│
├── Evaluation/
│   ├── bhagavad_gita_retrieval.json
│   └── Results/
│
├── Scripts/
│   ├── vedavault_chat_smoke.py
│   └── evaluation / indexing / utility scripts
│
├── Tests/
│   ├── test_api.py
│   ├── test_application.py
│   ├── test_conversation.py
│   ├── test_groq.py
│   ├── test_llm.py
│   ├── multilingual / retrieval / grounding tests
│   └── ...
│
├── README.md
└── .gitignore
