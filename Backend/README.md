# VedaVault Retrieval

`vedavault_retrieval` owns the application's retrieval contracts. It does not
depend on LangChain or LlamaIndex.

## Evidence bundles

`Retriever` finds ranked evidence. `EvidenceBundle` transports that evidence
unchanged in a stable, model-independent form: verse identity, text layer,
score, source, and immutable metadata remain available for traceability. A
future grounding layer will turn a bundle into model context, and a future LLM
will synthesize an answer from that context. Neither future layer is part of
this package yet.

Install the optional local retrieval dependencies when building an index:

```powershell
python -m pip install -r Backend/requirements-retrieval.txt
```

## Embeddings

The default is `intfloat/multilingual-e5-small`, an MIT-licensed, Sentence
Transformers-compatible retrieval model. It is selected for the first local
index because VedaVault requires cross-language retrieval across Sanskrit,
Hindi, Latin transliteration, and English, while the prior MiniLM default was
English-oriented. Multilingual E5 supports the 100 XLM-R languages and its
model card instructs retrieval clients to prefix documents with `passage: ` and
queries with `query: `; the provider applies those prefixes automatically.
Those prefixes belong to the explicitly selected E5 prompt profile, not to the
Sentence Transformers adapter universally. When selecting another model with
`VEDAVAULT_EMBEDDING_MODEL`, set `VEDAVAULT_EMBEDDING_PROMPT_PROFILE=none`
for a model that requires no role prefix (or pass a custom profile when using
the provider in code), and rebuild the index.

The model produces 384-dimensional vectors, has 12 layers and 0.1B parameters.
The published `model.safetensors` file is about 471 MB. Plan for roughly 1-2 GB
of available RAM for a CPU Python process after framework/model overhead; the
resulting dense index is approximately 4 bytes x dimensions x document count.
Sanskrit is a low-resource language in the underlying multilingual coverage,
so this default is a practical starting point, not a quality guarantee. Evaluate
retrieval against a VedaVault Sanskrit/Hindi benchmark before treating it as a
production choice.

Alternatives considered:

| Model | License / dimensions | Local tradeoff |
| --- | --- | --- |
| `intfloat/multilingual-e5-base` | MIT, 768 | Higher-capacity XLM-R model, but its published weight file is about 1.11 GB and CPU/RAM cost is materially higher. Use when evaluation justifies it. |
| `BAAI/bge-m3` | MIT, 1024 | 8,192-token, multilingual dense/sparse/ColBERT model. It is substantially heavier and its extra retrieval modes are not used by this simple dense index. |
| `sentence-transformers/LaBSE` | Apache-2.0, 768 | Supports 109 languages and is strong for translated-pair matching, but Sentence Transformers documents it as weaker for general semantic similarity than translation matching. |

Set `VEDAVAULT_EMBEDDING_MODEL` to select a compatible model and
`VEDAVAULT_EMBEDDING_DEVICE` to choose a device. For CPU use, the provider
defaults to batches of 32 inputs and uses up to eight PyTorch threads (or the
available CPU count if lower). Sentence Transformers performs batching in one
`encode` call, allowing it to group similar-length inputs and avoid Python-level
setup for every small batch. Exact duplicate document texts are encoded once;
every document and its distinct metadata remain in the index.
Override them with `VEDAVAULT_EMBEDDING_BATCH_SIZE` and
`VEDAVAULT_EMBEDDING_CPU_THREADS`; lower either value on constrained machines.
The model's native sequence limit remains in effect by default. Set
`VEDAVAULT_EMBEDDING_MAX_SEQ_LENGTH` only for a deliberately evaluated local
development profile (for example, `256`); it trades context for speed and must
match when rebuilding and querying an index.
Changing a model or embedding configuration requires rebuilding the local index.
Each local index persists a versioned `index_manifest.json` with its model,
vector dimension, prompt profile/prefixes, normalization setting, and sequence
length. Loading it with an incompatible provider fails before retrieval rather
than silently mixing vector semantics.

Build the first local index and run the demo from the repository root:

```powershell
python Scripts/build_bhagavad_gita_index.py
python Scripts/retrieval_demo.py
```

The index is a local NumPy `.npz` file plus JSON document metadata. It can be
replaced later without changing document, chunking, filtering, or retrieval
interfaces.

## Sources

Model claims and use instructions are taken from the official model cards and
Sentence Transformers documentation: [Multilingual E5 small](https://huggingface.co/intfloat/multilingual-e5-small),
[Multilingual E5 base](https://huggingface.co/intfloat/multilingual-e5-base),
[BGE-M3](https://huggingface.co/BAAI/bge-m3), and
[Sentence Transformers multilingual models](https://sbert.net/docs/sentence_transformer/pretrained_models.html).
