import numpy as np
from datasets import load_dataset
import json
from loguru import logger
from tqdm import tqdm
from typing import List, Dict, Any

# Load the query embeddings from a file
def load_query_embeddings(embedding_file: str) -> Dict[str, Any]:
    logger.info(f"Loading query embeddings from {embedding_file}...")
    embeddings = {}
    with open(embedding_file, "r") as f:
        for line in f:
            embedding = json.loads(line)
            embeddings[embedding["question"]] = embedding["embedding"]
    logger.info(f"Loaded {len(embeddings)} query embeddings.")
    return embeddings

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

# Load the query embeddings
embedding_file = "query_embeddings.jsonl"
query_embeddings_data = load_query_embeddings(embedding_file)

# Convert to arrays for processing
query_embeddings = np.array([query_embeddings_data[pq] for pq in processed_queries if pq in query_embeddings_data])
questions = [pq for pq in processed_queries if pq in query_embeddings_data]

# Load the Cohere multilingual dataset
logger.info("Loading Cohere multilingual dataset...")
cohere_dataset = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3", "en", split="train", streaming=True)

# Collect documents and their embeddings
docs = []
doc_embeddings = []

logger.info("Collecting documents and embeddings...")
for doc in tqdm(cohere_dataset, desc="Loading embeddings", unit="doc"):
    docs.append(doc)
    doc_embeddings.append(doc['emb'])
doc_embeddings = np.asarray(doc_embeddings)

# Function to perform brute force search
def brute_force_search(query_embedding: np.ndarray, doc_embeddings: np.ndarray, top_k: int) -> List[int]:
    dot_scores = np.matmul(query_embedding, doc_embeddings.transpose())[0]
    top_k_hits = np.argpartition(dot_scores, -top_k)[-top_k:].tolist()
    top_k_hits.sort(key=lambda x: dot_scores[x], reverse=True)
    return top_k_hits

# Perform search and collect results
num_selected = 20  # Number of top documents to retrieve
results = []

logger.info("Performing brute force search...")
for question, query_embedding in tqdm(zip(questions, query_embeddings), total=len(query_embeddings), desc="Processing queries", unit="query"):
    top_k_hits = brute_force_search(query_embedding, doc_embeddings, num_selected)
    
    query_results = []
    for rank, doc_id in enumerate(top_k_hits, start=1):
        query_results.append({
            "rank": rank,
            "title": docs[doc_id]['title'],
            "paragraph": docs[doc_id]['text'],
            "score": float(np.dot(query_embedding, doc_embeddings[doc_id].T))
        })
    results.append({"question": question, "results": query_results})

# Write results to a file
results_file = "search_results.json"
logger.info(f"Writing results to {results_file}...")
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results written to {results_file}")

