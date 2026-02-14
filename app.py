import json
import faiss
import numpy as np
import torch
import re
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===============================
# Configuration & Logging
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bfsi_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# System Paths

DATA_INDEX = "data/dataset_index.faiss"
DATA_TEXTS = "data/dataset_texts.json"
RAG_INDEX = "data/rag_index.faiss"
RAG_TEXTS = "data/rag_texts.json"

BASE_MODEL = "unsloth/tinyllama-chat-bnb-4bit"
LORA_PATH = "model/tinyllama_bfsi_lora"


# Thresholds

DATASET_SIMILARITY_THRESHOLD = 0.55
RAG_SIMILARITY_THRESHOLD = 0.65

RAG_TRIGGER_KEYWORDS = [
    "calculate", "formula", "policy", "policies",
    "interest", "penalty", "foreclosure",
    "insurance", "kyc", "emi", "how is", "explain"
]


# Guardrails

UNSAFE_KEYWORDS = [
    "hack", "steal", "fraud", "bypass", "scam",
    "illegal", "launder", "evade", "cheat"
]

OUT_OF_DOMAIN_KEYWORDS = [
    "weather", "recipe", "movie", "game",
    "sports", "celebrity", "politics"
]

SENSITIVE_DATA_PATTERNS = [
    r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    r'\b\d{9,16}\b',
]


# Load Models & Indexes

logger.info("Initializing BFSI Assistant...")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

dataset_index = faiss.read_index(DATA_INDEX)
with open(DATA_TEXTS, "r", encoding="utf-8") as f:
    dataset = json.load(f)

rag_index = faiss.read_index(RAG_INDEX)
with open(RAG_TEXTS, "r", encoding="utf-8") as f:
    rag_docs = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

logger.info("System ready!")


# Guardrail Functions

def validate_input(query):
    if not query or len(query.strip()) < 2:
        return False, "Please provide a valid question."

    q = query.lower()

    for word in UNSAFE_KEYWORDS:
        if word in q:
            return False, "I cannot assist with that request."

    out_count = sum(1 for k in OUT_OF_DOMAIN_KEYWORDS if k in q)
    if out_count >= 2:
        return False, "Please ask banking-related questions."

    return True, ""

def sanitize_text(text):
    for pattern in SENSITIVE_DATA_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text)
    return text

# Tier 1: Dataset Match

def get_dataset_match(query):
    emb = embed_model.encode([query], normalize_embeddings=True)
    D, I = dataset_index.search(np.array(emb), k=1)

    score = D[0][0]
    idx = I[0][0]

    if score < DATASET_SIMILARITY_THRESHOLD:
        logger.info(f"Tier 1 match (score: {score:.3f})")
        return dataset[idx]["output"]

    return None


def generate_text(prompt, max_tokens=60):
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=2.0,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        stop_markers = [
            "\n",
            "User:", "Assistant:",
            "Response:", "Question:",
            "Context:", "###"
        ]

        for marker in stop_markers:
            if marker in text:
                text = text.split(marker)[0]

        text = text.strip()

        if "." in text:
            text = text.split(".")[0] + "."

        if len(text) < 5:
            return "Please contact customer support for assistance."

        return text

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return "Please contact customer support for assistance."

def generate_slm_response(query):
    prompt = f"Question: {query}\nAnswer:"
    return generate_text(prompt, max_tokens=50)

def is_complex_query(query):
    q = query.lower()
    return any(word in q for word in RAG_TRIGGER_KEYWORDS)

def retrieve_rag_context(query, k=2):
    emb = embed_model.encode([query], normalize_embeddings=True)
    D, I = rag_index.search(np.array(emb), k=k)

    contexts = []
    for dist, idx in zip(D[0], I[0]):
        if dist < RAG_SIMILARITY_THRESHOLD:
            contexts.append(rag_docs[idx])

    return "\n\n".join(contexts)

def generate_rag_response(query, context):
    prompt = f"""Context: {context}

Question: {query}

Answer:"""
    return generate_text(prompt, max_tokens=70)

# Main Chatbot Logic

def chatbot(query):
    logger.info(f"Query: {query}")

    valid, msg = validate_input(query)
    if not valid:
        return f"[Security]\n{msg}"

    safe_query = sanitize_text(query)

    # Tier 1
    dataset_resp = get_dataset_match(safe_query)
    if dataset_resp:
        return f"[Tier 1: Dataset]\n{dataset_resp}"

    # Tier 3
    if is_complex_query(safe_query):
        context = retrieve_rag_context(safe_query)
        if context:
            rag_resp = generate_rag_response(safe_query, context)
            return f"[Tier 3: RAG]\n{rag_resp}"

    # Tier 2
    slm_resp = generate_slm_response(safe_query)
    return f"[Tier 2: SLM]\n{slm_resp}"


if __name__ == "__main__":
    print("=" * 60)
    print("BFSI Call Center AI Assistant")
    print("Type 'exit' to quit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break

            response = chatbot(user_input)
            print("\nAssistant:")
            print(response)

        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print("An error occurred. Please try again.")