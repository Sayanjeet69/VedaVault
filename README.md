 ॐ VedaVault

 Multilingual, Citation-Grounded AI for Indian Scripture

VedaVault is a multilingual Retrieval-Augmented Generation (RAG) application for exploring Indian scripture through natural-language conversation.

The current V1 focuses on the **Bhagavad Gita** and supports English, Hindi, Bengali, Sanskrit, transliterated Sanskrit, Hinglish, Banglish, and code-switched queries.

Unlike a normal LLM chatbot, VedaVault retrieves relevant scripture first and generates answers from supplied evidence with verse-level citations.

> **Ancient wisdom. Grounded answers. Modern clarity.**

Current V1 Capabilities:-
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

Architecture:-
                    Angular Frontend
                           │
                           │ REST / JSON
                           ▼
                       FastAPI
                           │
                    Session Context
                           │
                           ▼
                 Language Policy
                           │
                           ▼
              Query Understanding Provider
                           │
                           ▼
                        Retriever
                           │
                           ▼
                    EvidenceBundle
                           │
                           ▼
                  Evidence Hygiene
                           │
                           ▼
                   GroundingContext
                           │
                           ▼
                  GenerationRequest
                           │
                           ▼
                      LLMProvider
                    Qwen via Groq
                           │
                           ▼
                    AnswerContract
                           │
                           ▼
              Validated Grounded Response

Project Structure:-
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
│       ├── retriever / retrieval-related modules
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
| Tech                                            |                              Purpose in VedaVault |
|---|---|
| **Angular 20**                                  | Frontend UI, routing, responsive chat experience |
| **TypeScript**                                  | Strongly typed frontend logic and API models |
| **Tailwind CSS**                                | Styling, responsive layout, black/saffron theme |
| **Lucide Angular**                              | Clean UI icons |
| **FastAPI**                                     | HTTP backend exposing `/health`, `/answer`, session reset |
| **Python**                                      | Core backend/RAG implementation |
| **Uvicorn**                                     | Runs the FastAPI server |
| **Groq API**                                    | Hosted LLM inference |
| **Qwen 3.6 27B**                                | Query rewriting + grounded answer generation |
| **multilingual-e5-small**                       | Multilingual semantic embeddings for retrieval |
| **Custom vector retrieval/index**               | Finds relevant Bhagavad Gita passages |
| **Custom RAG pipeline**                         | `Retriever → EvidenceBundle → GroundingContext → AnswerContract` |
| **Evidence hygiene layer**                      | Prevents contaminated commentary from being treated as scripture translation |
| **Custom language policy**                      | Handles English, Hindi, Bengali, Sanskrit, code-switching, transliteration |
| **Conversation/session layer**                  | Maintains bounded multi-turn context and language continuity |
| **UUID sessions**                               | Safe opaque session IDs |
| **JSON / REST API**                             | Data exchange between Angular and FastAPI |
| **CORS**                                        | Allows the Angular frontend to securely call the backend |
| **Git + GitHub**                                | Version control, checkpoints, remote repository |
| **Unit tests**                                  | Backend/API/session/RAG/frontend regression protection |
| **Angular test tools**                          | Frontend interaction and API-service testing |
| **npm / Node.js**                               | Frontend dependency management and builds |

Limitations:-
- V1 currently contains only the Bhagavad Gita.
- Questions requiring wider Mahabharata context may not have sufficient evidence.
- Sessions are currently in-memory and reset when the backend restarts.
- Responses are request/response rather than token-streamed.
- Rich citation cards currently expose verse IDs; full Sanskrit/translation/provenance cards are still being expanded.
- Current retrieval is optimized for the V1 corpus rather than hundreds of texts.

Future Scope:-
Future versions are intended to expand VedaVault beyond the Bhagavad Gita into a broader Indian-scripture research platform.
Planned areas include:
- many more scriptures and philosophical texts
- generic multi-corpus architecture
- richer source and provenance exploration
- hybrid retrieval and reranking
- comparative analysis across scriptures/commentators
- additional Indian languages
- local/offline model support
- persistent research/session features
- multimodal document and manuscript support
The long-term goal is to make Bhagavad Gita V1 the first corpus, rather than a permanent architectural limitation.
