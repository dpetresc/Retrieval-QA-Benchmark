import json
from collections import defaultdict
from datasets import load_dataset

# Paths to your result files
faiss_result_file = 'results/mmlu_all/5_mmlu_unknown_model.jsonl'

# Load the original dataset
dataset = load_dataset('cais/mmlu', 'all', split='test')

# Create a mapping from question to subject
question_to_subject = {entry['question']: entry['subject'] for entry in dataset}

# Function to read and parse results
def parse_results(file_path):
    results = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Skip the first two lines which contain general statistics
    for line in lines[2:]:
        result = json.loads(line)
        results[result['id']] = result
    
    return results

# Parse results for FAISS models
faiss_results = parse_results(faiss_result_file)

key_prefix = 'no-index'

# Initialize dictionaries to store statistics
category_stats = defaultdict(lambda: {f'{key_prefix}_total': 0, f'{key_prefix}_correct': 0})
questions_per_category = defaultdict(list)

# Function to update statistics
def update_statistics(results, key_prefix):
    for result_id, result in results.items():
        question = result['question']
        generated = result['generated'].strip()
        answer = result['answer']
        choices = result['choices']
        matched = result['matched']
        
        # Retrieve the category (subject) from the original dataset
        subject = question_to_subject.get(question, 'unknown')
        
        # Update statistics
        category_stats[subject][f'{key_prefix}_total'] += 1
        if matched:
            category_stats[subject][f'{key_prefix}_correct'] += 1
        
        # Store questions per category for detailed analysis
        questions_per_category[subject].append(result)

update_statistics(faiss_results, 'no-index')

# Compute accuracy per category and print the comparison
for category, stats in category_stats.items():
    faiss_total = stats[f'{key_prefix}_total']
    faiss_correct = stats[f'{key_prefix}_correct']
    
    faiss_accuracy = faiss_correct / faiss_total if faiss_total > 0 else 0
    
    print(f"Category: {category}")
    print(f"Accuracy: {faiss_accuracy:.2%}\n")

# Compute overall accuracy
total_faiss_correct = sum(stats[f'{key_prefix}_correct'] for stats in category_stats.values())
total_faiss_total = sum(stats[f'{key_prefix}_total'] for stats in category_stats.values())

overall_faiss_accuracy = total_faiss_correct / total_faiss_total if total_faiss_total > 0 else 0

print(f"Overall Accuracy: {overall_faiss_accuracy:.2%}")
