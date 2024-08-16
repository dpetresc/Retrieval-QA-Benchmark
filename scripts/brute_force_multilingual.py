import numpy as np
import pyarrow.dataset as ds
import json
from loguru import logger
from tqdm import tqdm
from typing import List, Dict, Any
import glob
import os
from datasets import load_dataset
import time

# Load the query embeddings from a file
def load_query_embeddings(embedding_file: str) -> Dict[str, Any]:
    logger.info(f"Loading query embeddings from {embedding_file}...")
    start_time = time.time()
    
    embeddings = {}
    with open(embedding_file, "r") as f:
        for line in f:
            embedding = json.loads(line)
            embeddings[embedding["question"]] = embedding["embedding"]
    
    logger.info(f"Loaded {len(embeddings)} query embeddings in {time.time() - start_time:.2f} seconds.")
    return embeddings

# Preprocess MMLU queries
def preproc_question4query(data: Dict[str, Any]) -> str:
    question = data["id"]
    choices = " | ".join([data["option_a"], data["option_b"], data["option_c"], data["option_d"]])
    return "\n".join([question, choices])

# Load and preprocess the MMLU dataset for a specific language
def load_and_preprocess_dataset(language: str) -> List[str]:
    logger.info(f"Loading MMLU dataset for language: {language}...")
    dataset = load_dataset('alexandrainst/m_mmlu', language, split='test')
    
    queries = [{"id": d["id"], "option_a": d["option_a"], "option_b": d["option_b"], "option_c": d["option_c"], "option_d": d["option_d"]} for d in dataset]
    processed_queries = [preproc_question4query(query) for query in queries]
    
    logger.info(f"Number of processed queries for {language}: {len(processed_queries)}")
    return processed_queries

# Perform brute force search on all queries
def brute_force_search(query_embeddings: np.ndarray, doc_embeddings: np.ndarray, docs: List[Dict[str, Any]], ids: List[str], top_k: int) -> List[List[Dict[str, Any]]]:
    # Compute dot products between all queries and all documents
    dot_scores = np.matmul(query_embeddings, doc_embeddings.transpose())
    
    all_results = []
    
    # For each query, find the top-k documents
    for query_idx in range(dot_scores.shape[0]):
        query_scores = dot_scores[query_idx]
        top_k_hits = np.argpartition(query_scores, -top_k)[-top_k:].tolist()
        top_k_hits.sort(key=lambda x: query_scores[x], reverse=True)
        
        top_k_results = [
            {
                "rank": rank + 1,
                "id": ids[doc_id],
                "title": docs[doc_id]['title'],
                "paragraph": docs[doc_id]['text'],
                "score": query_scores[doc_id],
            }
            for rank, doc_id in enumerate(top_k_hits)
        ]
        
        all_results.append(top_k_results)
    
    return all_results

def merge_top_k(existing_results: List[Dict[str, Any]], new_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    combined_results = existing_results + new_results
    combined_results.sort(key=lambda x: x["score"], reverse=True)

    # Re-rank after merging
    for idx, result in enumerate(combined_results[:top_k]):
        result["rank"] = idx + 1

    return combined_results[:top_k]

# Function to save intermediate results to a file
def save_intermediate_results(results: Dict[str, List[Dict[str, Any]]], results_file: str) -> None:
    logger.info(f"Saving intermediate results to {results_file}...")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {results_file}")

# Main processing function
def process_language(language: str, embedding_file: str, results_file: str, processed_files_file: str, parquet_files_path: str):
    # Load queries and embeddings
    queries = load_and_preprocess_dataset(language)
    query_embeddings_data = load_query_embeddings(embedding_file)

    # Convert to arrays for processing
    query_embeddings = np.array([query_embeddings_data[pq] for pq in queries if pq in query_embeddings_data])
    questions = [pq for pq in queries if pq in query_embeddings_data]

    # Load intermediate results if available
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = {q: [] for q in questions}

    # Load the list of processed files
    if os.path.exists(processed_files_file):
        with open(processed_files_file, "r") as f:
            processed_files = set(f.read().splitlines())
    else:
        processed_files = set()

    # Perform search on Parquet files
    parquet_files = sorted(glob.glob(parquet_files_path), key=sort_key)
    num_selected = 20  # Number of top documents to retrieve for each query

    for parquet_file in tqdm(parquet_files, desc=f"Processing {language.upper()} Parquet files", unit="file"):
        # Skip already processed files
        if parquet_file in processed_files:
            logger.info(f"Skipping already processed file: {parquet_file}")
            continue

        logger.info(f"Processing {parquet_file}...")
        file_start_time = time.time()

        # Load the Parquet file as a dataset
        dataset = ds.dataset(parquet_file, format="parquet")

        # Load all data in one go (no batching)
        table = dataset.to_table()

        doc_embeddings = np.vstack(table.column("emb").to_pylist())  # Convert list of embeddings to 2D numpy array
        ids = table.column("_id").to_pylist()  # Convert Arrow array to Python list
        docs = [{"title": title.as_py(), "text": text.as_py()} for title, text in zip(table.column("title"), table.column("text"))]

        # Process all queries for the language
        results_batch = brute_force_search(query_embeddings, doc_embeddings, docs, ids, num_selected)
        for i, question in enumerate(questions):
            results[question] = merge_top_k(results[question], results_batch[i], num_selected)

        logger.info(f"Processed {parquet_file} in {time.time() - file_start_time:.2f} seconds")

        # Save intermediate results after processing each Parquet file
        save_intermediate_results(results, results_file)

        # Mark this file as processed and save the progress
        processed_files.add(parquet_file)
        with open(processed_files_file, "w") as f:
            f.write("\n".join(processed_files))

    print(f"Results for {language.upper()} saved to {results_file}")
    print(f"Processed files list for {language.upper()} saved to {processed_files_file}")

# Utility function to extract the numerical part from the filename for correct sorting
def sort_key(file_path):
    # Extract the base name and remove the .parquet extension
    base_name = os.path.basename(file_path)
    # Extract the numeric part from the base name
    number = int(base_name.split('.')[0])
    return number

# Process both German and Dutch datasets
process_language(
    language="de",
    embedding_file="german_query_embeddings.jsonl",
    results_file="results/german_search_results.json",
    processed_files_file="results/processed_files_de.txt",
    parquet_files_path="index/wikidataset_de/*.parquet"
)

process_language(
    language="nl",
    embedding_file="dutch_query_embeddings.jsonl",
    results_file="results/dutch_search_results.json",
    processed_files_file="results/processed_files_nl.txt",
    parquet_files_path="index/wikidataset_nl/*.parquet"
)

