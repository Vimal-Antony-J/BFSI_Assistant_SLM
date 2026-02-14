import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/bfsi_alpaca_dataset_.json"
INDEX_PATH = "data/dataset_index.faiss"
TEXTS_PATH = "data/dataset_texts.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

with open(DATA_PATH, "r") as f:
    dataset = json.load(f)

texts = [item["instruction"] for item in dataset]
embeddings = model.encode(texts)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, INDEX_PATH)

with open(TEXTS_PATH, "w") as f:
    json.dump(dataset, f)

print("Dataset similarity index built.")