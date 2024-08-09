import numpy as np
from datasets import load_dataset
import cohere
import json
from loguru import logger
from tqdm import tqdm
from typing import Dict, Any, List, Optional
import os
import time

# Function to read API key from a file
def read_api_key(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read().strip()

# Read the Cohere API key from a file
api_key_file = "../index/cohere_api_key.txt"
COHERE_API_KEY = read_api_key(api_key_file)
co = cohere.Client(COHERE_API_KEY)

# Preprocess MMLU queries
def preproc_question4query(data: Dict[str, Any]) -> str:
    question = data["question"]
    choices = " | ".join(data["choices"])
    return "\n".join([question, choices])

# Load the MMLU dataset
logger.info("Loading MMLU dataset...")
dataset = load_dataset('cais/mmlu', 'all', split='test')

print("Number of rows in dataset:", len(dataset))

# Prepare queries
queries = [{"question": d["question"], "choices": d["choices"]} for d in dataset]
processed_queries = [preproc_question4query(query) for query in queries]

print("NB processed queries ", len(processed_queries))

# Function to load existing embeddings
def load_existing_embeddings(embedding_file: str) -> Dict[str, Any]:
    if os.path.exists(embedding_file):
        logger.info(f"Loading existing query embeddings from {embedding_file}...")
        embeddings = {}
        with open(embedding_file, "r") as f:
            for line in f:
                embedding = json.loads(line)
                embeddings[embedding["question"]] = embedding["embedding"]
        logger.info(f"Loaded {len(embeddings)} existing query embeddings.")
        return embeddings
    return {}

# Function to append new embeddings to the file
def append_embeddings_to_file(embedding_file: str, new_embeddings: Dict[str, Any]) -> None:
    with open(embedding_file, "a") as f:
        for question, embedding in new_embeddings.items():
            f.write(json.dumps({"question": question, "embedding": embedding}) + "\n")

# Function to encode queries in batches and append to file
def encode_queries_in_batches(processed_queries: List[str], model: str, embedding_file: str, batch_size: int = 100, delay: int = 60) -> None:
    existing_embeddings = load_existing_embeddings(embedding_file)
    start_index = len(existing_embeddings)
    logger.info(f"Starting index: {start_index}")

    remaining_queries = [q for q in processed_queries if q not in existing_embeddings]
    remaining_batches = len(remaining_queries) // batch_size + (1 if len(remaining_queries) % batch_size != 0 else 0)
    
    for i in tqdm(range(0, len(remaining_queries), batch_size), total=remaining_batches, desc="Calculating embeddings", unit="batch"):
        batch_queries = remaining_queries[i:i + batch_size]

        logger.info(f"Processing batch starting at index {i}, batch size: {len(batch_queries)}")

        response = co.embed(texts=batch_queries, model=model, input_type="search_query")
        if len(response.embeddings) != len(batch_queries):
            logger.error(f"Mismatch in number of embeddings received: expected {len(batch_queries)}, got {len(response.embeddings)}")
            continue

        batch_embeddings = response.embeddings

        new_embeddings = {q: emb for q, emb in zip(batch_queries, batch_embeddings)}

        # Append new embeddings to the file
        append_embeddings_to_file(embedding_file, new_embeddings)
        logger.info(f"Appended {len(new_embeddings)} new embeddings to {embedding_file}")

        time.sleep(delay)  # Respect rate limit

# Encode queries
embedding_file = "query_embeddings.jsonl"  # Use .jsonl for line-delimited JSON
encode_queries_in_batches(processed_queries, 'embed-multilingual-v3.0', embedding_file)

print(f"Embeddings saved to {embedding_file}")

