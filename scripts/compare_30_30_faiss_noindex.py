import json
from collections import defaultdict
from datasets import load_dataset

# Paths to your result files
faiss_result_file_30tokens = 'results/mmlu_all/5_mmlu-faiss_unknown_model_30tokens.jsonl'
unknown_model_30tokens = 'results/mmlu_all/5_mmlu_unknown_model_30tokens.jsonl'

# Load the original dataset
dataset = load_dataset('cais/mmlu', 'all', split='test')

# Create a mapping from question to subject
question_to_subject = {entry['question']: entry['subject'] for entry in dataset}

# Function to read and parse results, skipping the first two lines
def parse_results(file_path):
    results = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines[2:]:  # Skip the first two lines
        result = json.loads(line)
        results[result['id']] = result
    
    return results

# Parse results for both models
faiss_results_30tokens = parse_results(faiss_result_file_30tokens)
unknown_results_30tokens = parse_results(unknown_model_30tokens)

# Initialize dictionaries to store statistics for college_chemistry
category_stats_faiss = defaultdict(lambda: {'total': 0, 'correct': 0})
category_stats_unknown = defaultdict(lambda: {'total': 0, 'correct': 0})

# Function to update statistics for a specific category
def update_statistics(results, category_stats, target_subject):
    for result_id, result in results.items():
        question = result['question']
        generated = result['generated'].strip()
        matched = result['matched']
        
        # Retrieve the category (subject) from the original dataset
        subject = question_to_subject.get(question, 'unknown')
        
        if subject == target_subject:
            # Update statistics for the target subject
            category_stats[subject]['total'] += 1
            if matched:
                category_stats[subject]['correct'] += 1

# Update statistics for both models for college_chemistry
target_subject = 'college_chemistry'
update_statistics(faiss_results_30tokens, category_stats_faiss, target_subject)
update_statistics(unknown_results_30tokens, category_stats_unknown, target_subject)

# Compute accuracy for college_chemistry
faiss_total = category_stats_faiss[target_subject]['total']
faiss_correct = category_stats_faiss[target_subject]['correct']
faiss_accuracy = faiss_correct / faiss_total if faiss_total > 0 else 0

unknown_total = category_stats_unknown[target_subject]['total']
unknown_correct = category_stats_unknown[target_subject]['correct']
unknown_accuracy = unknown_correct / unknown_total if unknown_total > 0 else 0

print(f"Category: {target_subject}")
print(f"FAISS 30 Tokens - Accuracy: {faiss_accuracy:.2%} ({faiss_correct}/{faiss_total})")
print(f"Unknown Model 30 Tokens - Accuracy: {unknown_accuracy:.2%} ({unknown_correct}/{unknown_total})")

# Check and print full_output when one matched but the other didn't
for result_id in faiss_results_30tokens:
    if result_id in unknown_results_30tokens:
        matched_faiss = faiss_results_30tokens[result_id]['matched']
        matched_unknown = unknown_results_30tokens[result_id]['matched']
        
        if matched_faiss != matched_unknown:
            print(f"\nQuestion ID: {result_id}")
            print(f"Question: {faiss_results_30tokens[result_id]['question']}")
            print(f"Choices: {faiss_results_30tokens[result_id]['choices']}")
            print(f"Correct Answer: {faiss_results_30tokens[result_id]['answer']}")
            print(f"FAISS 30 Tokens - Generated Answer: {faiss_results_30tokens[result_id]['generated'].strip()}")
            print(f"FAISS 30 Tokens - Full Output:\n{faiss_results_30tokens[result_id]['full_output']}")
            print(f"Unknown Model 30 Tokens - Generated Answer: {unknown_results_30tokens[result_id]['generated'].strip()}")
            print(f"Unknown Model 30 Tokens - Full Output:\n{unknown_results_30tokens[result_id]['full_output']}")

