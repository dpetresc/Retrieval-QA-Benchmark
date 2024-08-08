from datasets import load_dataset
import random

# Set a random seed for reproducibility
random.seed(42)

# Load the original dataset
dataset = load_dataset('cais/mmlu', 'all', split='test')

# Create a mapping from question to subject
question_to_subject = {entry['question']: entry['subject'] for entry in dataset}

# Calculate the number of questions to sample (10% of each category)
subject_counts = {}
for entry in dataset:
    subject = entry['subject']
    if subject not in subject_counts:
        subject_counts[subject] = 0
    subject_counts[subject] += 1

subject_sample_counts = {subject: max(1, int(count * 0.5)) for subject, count in subject_counts.items()}

# Sample 10% of each category
sampled_questions = {}
for subject in subject_sample_counts:
    subject_questions = [entry for entry in dataset if entry['subject'] == subject]
    sampled_questions[subject] = random.sample(subject_questions, subject_sample_counts[subject])

# Save the sampled questions to a new dataset
sampled_dataset = [question for questions in sampled_questions.values() for question in questions]

# Write the sampled questions to a file
with open('sampled_questions.txt', 'w') as f:
    for entry in sampled_dataset:
        f.write(f"Question: {entry['question']}\n")
        f.write(f"Subject: {entry['subject']}\n")
        f.write("\n")

print(f"Total sampled questions: {len(sampled_dataset)}")
print("Sampled questions have been written to 'sampled_questions.txt'")

