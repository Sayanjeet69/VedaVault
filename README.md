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

Tech Stack:-
| Technology | Purpose in VedaVault |
|---|---|
| **Angular 20** | Frontend UI, routing and responsive chat experience |
| **TypeScript** | Strongly typed frontend logic and API models |
| **Tailwind CSS** | Styling, responsive layout and black/saffron theme |
| **Lucide Angular** | Lightweight UI icons |
| **FastAPI** | HTTP backend exposing `/health`, `/answer` and session reset |
| **Python** | Core backend and RAG implementation |
| **Uvicorn** | Runs the FastAPI server |
| **Groq API** | Hosted LLM inference |
| **Qwen 3.6 27B** | Query rewriting and grounded answer generation |
| **multilingual-e5-small** | Multilingual semantic embeddings |
| **Custom vector retrieval/index** | Finds relevant Bhagavad Gita passages |
| **Custom RAG pipeline** | Retrieval, grounding, generation and answer validation |
| **Evidence hygiene layer** | Prevents contaminated commentary from being treated as scripture translation |
| **Custom language policy** | Handles English, Hindi, Bengali, Sanskrit, code-switching and transliteration |
| **Conversation/session layer** | Maintains bounded multi-turn context and language continuity |
| **UUID sessions** | Safe opaque session identifiers |
| **JSON / REST API** | Angular ↔ FastAPI communication |
| **CORS** | Controlled frontend/backend communication |
| **Git + GitHub** | Version control and project history |
| **Unit tests** | Backend/API/session/RAG regression protection |
| **Angular test tools** | Frontend interaction and API-service testing |
| **npm / Node.js** | Frontend dependency management and builds |

Data Sources:-
V1 uses a structured Bhagavad Gita corpus built from multiple CSV/JSON sources containing:
- all 701 canonical verses
- Sanskrit text
- English translations
- Hindi translations
- multiple translation sources
- commentary/provenance metadata
During corpus auditing, some records labelled as translations were found to contain commentary. VedaVault therefore uses an Evidence Hygiene layer to ensure only approved clean evidence reaches the grounding context.

Key Engineering Decisions:-
- Custom RAG architecture instead of framework-heavy orchestration
- Model-independent LLM provider boundary
- Conversation history is context, never scripture evidence
- Fresh retrieval on every turn
- Evidence provenance separated from grounding provenance
- Fail-closed behavior when evidence is insufficient
- Model memory is not allowed to substitute for retrieved scripture
- Frontend and backend kept independently modular

Limitations:-
- V1 currently contains only the Bhagavad Gita.
- Questions requiring wider Mahabharata context may not have sufficient evidence.
- Sessions are currently in-memory and reset when the backend restarts.
- Responses are request/response rather than token-streamed.
- Rich citation cards currently expose verse IDs; full Sanskrit/translation/provenance cards are still being expanded.
- Current retrieval is optimized for the V1 corpus rather than hundreds of texts.

Future Scope:-
Future versions are intended to expand VedaVault into a broader Indian-scripture research platform.
Planned areas include:
- many more scriptures and philosophical texts
- generic multi-corpus architecture
- richer source and provenance exploration
- hybrid retrieval and reranking
- comparative analysis across scriptures and commentators
- additional Indian languages
- local/offline model support
- persistent research/session features
- multimodal document and manuscript support
The long-term goal is to make the Bhagavad Gita V1 the first corpus, rather than a permanent architectural limitation.

Current Status:-
VedaVault V1 is a complete local working demonstrator with:
- custom multilingual RAG
- grounded generation
- citation-aware answers
- conversation sessions
- FastAPI backend
- Angular frontend
- desktop/mobile responsive UI
- live frontend ↔ backend integration
Public deployment is the next major milestone.
