import json
from collections import defaultdict
from datasets import load_dataset

# Paths to your result files
faiss_result_file_3tokens = 'results/mmlu_all/5_mmlu-faiss_unknown_model_3tokens.jsonl'
faiss_result_file_30tokens = 'results/mmlu_all/5_mmlu-faiss_unknown_model_30tokens.jsonl'

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


# Parse results for both FAISS models
faiss_results_3tokens = parse_results(faiss_result_file_3tokens)
faiss_results_30tokens = parse_results(faiss_result_file_30tokens)

# Initialize dictionaries to store statistics for college_chemistry
category_stats_3tokens = defaultdict(lambda: {'total': 0, 'correct': 0})
category_stats_30tokens = defaultdict(lambda: {'total': 0, 'correct': 0})

# Function to update statistics for a specific category
def update_statistics(results, category_stats, target_subject):
    for result_id, result in results.items():
        question = result['question']
        generated = result['generated'].strip()
        answer = result['answer']
        choices = result['choices']
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
update_statistics(faiss_results_3tokens, category_stats_3tokens, target_subject)
update_statistics(faiss_results_30tokens, category_stats_30tokens, target_subject)

# Compute accuracy for college_chemistry
faiss_total_3tokens = category_stats_3tokens[target_subject]['total']
faiss_correct_3tokens = category_stats_3tokens[target_subject]['correct']
faiss_accuracy_3tokens = faiss_correct_3tokens / faiss_total_3tokens if faiss_total_3tokens > 0 else 0

faiss_total_30tokens = category_stats_30tokens[target_subject]['total']
faiss_correct_30tokens = category_stats_30tokens[target_subject]['correct']
faiss_accuracy_30tokens = faiss_correct_30tokens / faiss_total_30tokens if faiss_total_30tokens > 0 else 0

print(f"Category: {target_subject}")
print(f"3 Tokens - Accuracy: {faiss_accuracy_3tokens:.2%} ({faiss_correct_3tokens}/{faiss_total_3tokens})")
print(f"30 Tokens - Accuracy: {faiss_accuracy_30tokens:.2%} ({faiss_correct_30tokens}/{faiss_total_30tokens})")

# Check and print full_output when one matched but the other didn't
for result_id in faiss_results_3tokens:
    if result_id in faiss_results_30tokens:
        matched_3tokens = faiss_results_3tokens[result_id]['matched']
        matched_30tokens = faiss_results_30tokens[result_id]['matched']
        
        if matched_3tokens != matched_30tokens:
            print(f"\nQuestion ID: {result_id}")
            print(f"Question: {faiss_results_3tokens[result_id]['question']}")
            print(f"Golden answer: {faiss_results_30tokens[result_id]['answer']}")
            print(f"3 Tokens - Full Output:\n{faiss_results_3tokens[result_id]['full_output']}")
            print(f"30 Tokens - Full Output:\n{faiss_results_30tokens[result_id]['full_output']}")
            print(f"3 Tokens - Matched: {matched_3tokens}")
            print(f"30 Tokens - Matched: {matched_30tokens}")

