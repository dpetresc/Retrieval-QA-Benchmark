import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd
import numpy as np
import glob
import time
from tqdm import tqdm
from datasets import load_dataset

# Define the pattern to match the Arrow files
arrow_files_pattern = "../../coherepedia-jvector/Cohere___wikipedia-2023-11-embed-multilingual-v3/en/0.0.0/37feace541fadccf70579e9f289c3cf8e8b186d7/wikipedia-2023-11-embed-multilingual-v3-train-*.arrow"

# Get the list of Arrow files
arrow_files = sorted(glob.glob(arrow_files_pattern), reverse=True)

# Initialize an empty DataFrame
df = pd.DataFrame()

# Function to ensure 'emb' is a numpy array of floats if it's a list
def ensure_embedding(emb_column):
    return [np.array(emb, dtype=np.float32) if isinstance(emb, list) else emb for emb in emb_column]

i = 0

# Read each Arrow file and concatenate it into the DataFrame
for arrow_file in tqdm(arrow_files, desc="Processing Arrow Files", unit="file"):
    start_time = time.time()
    print(f"\nProcessing {arrow_file}...")
    
    # Open the Arrow file using the IPC Stream format
    with pa.memory_map(arrow_file, 'r') as source:
        reader = ipc.RecordBatchStreamReader(source)
        
        # Process each batch within the file
        for batch in tqdm(reader, desc="Reading Batches", unit="batch"):
            table = pa.Table.from_batches([batch])
            df_chunk = table.to_pandas()

            # Ensure 'emb' is a list of floats (numpy arrays)
            df_chunk['emb'] = ensure_embedding(df_chunk['emb'].tolist())
            #print(f"First 'emb' value after conversion: {df_chunk['emb'].iloc[0]}")  # Debugging info

            # Concatenate to the main DataFrame
            df = pd.concat([df, df_chunk], ignore_index=True)
    
    i += 1
    if i > 0:
        break
    
    # Calculate and print the time taken for this file
    end_time = time.time()
    print(f"Time taken to process {arrow_file}: {end_time - start_time:.2f} seconds")

# Print the number of rows in the DataFrame
print(f"Total number of rows in the DataFrame: {len(df)}")

# Print the first 5 rows
print("First 5 rows of the DataFrame:")
print(df.head())

# Set the maximum column width to display full content
pd.set_option('display.max_colwidth', None)

# Print the first row completely
print("First row of the DataFrame:")
print(df.tail(1).to_string(index=False))

lang = "en"  # Specify the language you want to use
docs = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3", lang, split="train", streaming=True)

# Initialize lists to store the first 5 elements from both sources
first_5_stream = []
first_5_arrow = []

# Read the first 5 elements from the streaming dataset
for i, doc in enumerate(docs):
    if i < 5:
        first_5_stream.append(doc['emb'])
    else:
        break  # Stop after getting the first 5
first_5_stream = np.array(first_5_stream)

# Read the first 5 elements from the Arrow files
for arrow_file in tqdm(arrow_files, desc="Processing Arrow Files", unit="file"):
    with pa.memory_map(arrow_file, 'r') as source:
        reader = ipc.RecordBatchStreamReader(source)
        for batch in reader:
            table = pa.Table.from_batches([batch])
            df_chunk = table.to_pandas()

            # Store the first 5 elements
            if len(first_5_arrow) < 5:
                first_5_arrow.extend(df_chunk['emb'].tolist()[:5-len(first_5_arrow)])
            
            if len(first_5_arrow) >= 5:
                break  # Stop once we have 5 elements
    if len(first_5_arrow) >= 5:
        break

first_5_arrow = np.array(first_5_arrow)

# Compare the first 5 elements
print("Comparing the first 5 elements...")
print("Streaming first 5:", first_5_stream)
print("Arrow first 5:", first_5_arrow)
first_5_comparison = np.isclose(first_5_stream, first_5_arrow)
print("Comparison result for the first 5 elements:", first_5_comparison)
print("Are all first 5 elements equal?", first_5_comparison.all())

# Optionally, print mismatches
if not first_5_comparison.all():
    mismatch_indices = np.where(first_5_comparison == False)[0]
    print("Mismatch at indices:", mismatch_indices)
    print("Streaming values at mismatches:", first_5_stream[mismatch_indices])
    print("Arrow values at mismatches:", first_5_arrow[mismatch_indices])

