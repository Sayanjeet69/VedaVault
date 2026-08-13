# VedaVault Retrieval

`vedavault_retrieval` owns the application's retrieval contracts. It does not
depend on LangChain or LlamaIndex.

Install the optional local retrieval dependencies when building an index:

```powershell
python -m pip install -r Backend/requirements-retrieval.txt
```

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`. Set
`VEDAVAULT_EMBEDDING_MODEL` to select another compatible Sentence Transformers
model, and optionally set `VEDAVAULT_EMBEDDING_DEVICE` (for example `cpu`).

Build the first local index and run the demo from the repository root:

```powershell
python Scripts/build_bhagavad_gita_index.py
python Scripts/retrieval_demo.py
```

The index is a local NumPy `.npz` file plus JSON document metadata. It can be
replaced later without changing document, chunking, filtering, or retrieval
interfaces.
