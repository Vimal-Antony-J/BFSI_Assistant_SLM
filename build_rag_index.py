import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

DOCS_DIR = "data/rag_docs"
INDEX_PATH = "data/rag_index.faiss"
DOC_STORE = "data/rag_texts.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
for file in os.listdir(DOCS_DIR):
    with open(os.path.join(DOCS_DIR, file), "r") as f:
        documents.append(f.read())

embeddings = model.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, INDEX_PATH)

with open(DOC_STORE, "w") as f:
    json.dump(documents, f)

print("RAG index built.")