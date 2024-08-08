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

# Prepare queries
queries = [{"question": d["question"], "choices": d["choices"]} for d in dataset]
processed_queries = [preproc_question4query(query) for query in queries]
questions = [query["question"] for query in queries]

# Function to load or calculate query embeddings in batches
def encode_queries_in_batches(queries: List[str], questions: List[str], model: str, embedding_file: str, batch_size: int = 100, delay: int = 60) -> List[Dict[str, Any]]:
    embeddings_data = []
    if os.path.exists(embedding_file):
        logger.info(f"Loading query embeddings from {embedding_file}...")
        with open(embedding_file, "r") as f:
            embeddings_data = json.load(f)

    start_index = len(embeddings_data)
    total_batches = (len(queries) - start_index) // batch_size + (1 if (len(queries) - start_index) % batch_size != 0 else 0)
    
    for i in tqdm(range(start_index, len(queries), batch_size), desc="Calculating embeddings", unit="batch"):
        batch_queries = queries[i:i + batch_size]
        batch_questions = questions[i:i + batch_size]
        response = co.embed(texts=batch_queries, model=model, input_type="search_query")
        batch_embeddings = response.embeddings
        
        for question, embedding in zip(batch_questions, batch_embeddings):
            embeddings_data.append({"question": question, "embedding": embedding})
        
        # Save progress every batch
        with open(embedding_file, "w") as f:
            json.dump(embeddings_data, f)
        
        time.sleep(delay)  # Respect rate limit

    return embeddings_data

# Encode queries
embedding_file = "query_embeddings.json"
query_embeddings = encode_queries_in_batches(processed_queries, questions, 'embed-multilingual-v3.0', embedding_file)

print(f"Embeddings saved to {embedding_file}")

