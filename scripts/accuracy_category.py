import json
from collections import defaultdict
from datasets import load_dataset

# Paths to your result files
faiss_result_file = 'results/mmlu_all/5_mmlu-faiss_unknown_model.jsonl'
non_faiss_result_file = 'results/mmlu_all/5_mmlu_unknown_model.jsonl'

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

# Parse results for both FAISS and non-FAISS models
faiss_results = parse_results(faiss_result_file)
non_faiss_results = parse_results(non_faiss_result_file)

# Initialize dictionaries to store statistics
category_stats = defaultdict(lambda: {'faiss_total': 0, 'faiss_correct': 0, 'non_faiss_total': 0, 'non_faiss_correct': 0})
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

# Update statistics for both FAISS and non-FAISS results
update_statistics(faiss_results, 'faiss')
update_statistics(non_faiss_results, 'non_faiss')

# Compute accuracy per category and print the comparison
for category, stats in category_stats.items():
    faiss_total = stats['faiss_total']
    faiss_correct = stats['faiss_correct']
    non_faiss_total = stats['non_faiss_total']
    non_faiss_correct = stats['non_faiss_correct']
    
    faiss_accuracy = faiss_correct / faiss_total if faiss_total > 0 else 0
    non_faiss_accuracy = non_faiss_correct / non_faiss_total if non_faiss_total > 0 else 0
    
    improvement = faiss_accuracy - non_faiss_accuracy
    
    print(f"Category: {category}")
    print(f"  FAISS Accuracy: {faiss_accuracy:.2%}")
    print(f"  Non-FAISS Accuracy: {non_faiss_accuracy:.2%}")
    print(f"  Improvement: {'+' if improvement >= 0 else ''}{improvement:.2%}\n")

# Compute overall accuracy
total_faiss_correct = sum(stats['faiss_correct'] for stats in category_stats.values())
total_faiss_total = sum(stats['faiss_total'] for stats in category_stats.values())
total_non_faiss_correct = sum(stats['non_faiss_correct'] for stats in category_stats.values())
total_non_faiss_total = sum(stats['non_faiss_total'] for stats in category_stats.values())

overall_faiss_accuracy = total_faiss_correct / total_faiss_total if total_faiss_total > 0 else 0
overall_non_faiss_accuracy = total_non_faiss_correct / total_non_faiss_total if total_non_faiss_total > 0 else 0

print(f"Overall FAISS Accuracy: {overall_faiss_accuracy:.2%}")
print(f"Overall Non-FAISS Accuracy: {overall_non_faiss_accuracy:.2%}")

## Optionally, write detailed results to a file
#with open('detailed_results_per_category_comparison.json', 'w') as outfile:
#    json.dump({
#        'category_stats': category_stats,
#        'questions_per_category': questions_per_category
#    }, outfile, indent=2)
#
## Print every question with its accuracy details
#for subject, questions in questions_per_category.items():
#    for question in questions:
#        question_text = question['question']
#        generated_faiss = faiss_results[question['id']]['generated'].strip() if question['id'] in faiss_results else 'N/A'
#        matched_faiss = faiss_results[question['id']]['matched'] if question['id'] in faiss_results else 0
#        generated_non_faiss = non_faiss_results[question['id']]['generated'].strip() if question['id'] in non_faiss_results else 'N/A'
#        matched_non_faiss = non_faiss_results[question['id']]['matched'] if question['id'] in non_faiss_results else 0
#        
#        print(f"Question: {question_text}")
#        print(f"  Subject: {subject}")
#        print(f"  FAISS Generated: {generated_faiss}")
#        print(f"  FAISS Matched: {matched_faiss}")
#        print(f"  Non-FAISS Generated: {generated_non_faiss}")
#        print(f"  Non-FAISS Matched: {matched_non_faiss}\n")
#
