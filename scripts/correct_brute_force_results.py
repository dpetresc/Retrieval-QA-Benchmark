import json
from loguru import logger
from typing import List, Dict, Any

# Function to verify and update rankings
def verify_and_update_rankings(results: Dict[str, List[Dict[str, Any]]]) -> None:
    for query, docs in results.items():
        # Verify that there are exactly 20 documents
        if len(docs) != 20:
            logger.warning(f"Query '{query}' does not have exactly 20 documents, it has {len(docs)} documents.")
            continue
        
        # Verify that the documents are sorted by score in descending order
        scores = [doc["score"] for doc in docs]
        if scores != sorted(scores, reverse=True):
            logger.warning(f"Documents for query '{query}' are not sorted by score in descending order.")
            continue

        # Update the rank based on the correct order
        for idx, doc in enumerate(docs):
            doc["rank"] = idx + 1

    logger.info("Rankings have been verified and updated where necessary.")

# Load the existing results from the JSON file
results_file = "./results/search_results.json"
with open(results_file, "r") as f:
    results = json.load(f)

# Verify and update rankings
verify_and_update_rankings(results)

# Save the updated results back to the JSON file
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)

logger.info(f"Updated results saved to {results_file}")

