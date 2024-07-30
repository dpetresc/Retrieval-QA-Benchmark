import json
import argparse

def compute_metrics(results):
    transform_times = []
    completion_times = []
    total_times = []
    ttft_times = []
    tpot_times = []
    prompt_tokens = []
    completion_tokens = []
    generated_tokens = []

    for result in results:
        profile_time = result.get("profile_time", {})
        transform_times.append(profile_time.get("transform.TransformChain.profile", 0) / 1000)  # Convert to seconds
        completion_times.append(profile_time.get("model.local.completion.profile", 0) / 1000)  # Convert to seconds
        total_times.append(profile_time.get("model.local-llm.profile", 0) / 1000)  # Convert to seconds

        ttft_times.append(profile_time.get("model.local.ttft.profile", 0) / 1000)  # Convert to seconds
        tpot_times.append(profile_time.get("model.local.tpot.profile", 0) / 1000)  # Convert to seconds

        prompt_tokens.append(result.get("prompt_tokens", 0))
        completion_tokens.append(result.get("completion_tokens", 0))

    def compute_average(lst):
        return sum(lst) / len(lst) if lst else 0

    metrics = {
        "transform_time_avg": compute_average(transform_times),
        "transform_time_max": max(transform_times, default=0),
        "transform_time_min": min(transform_times, default=0),
        "completion_time_avg": compute_average(completion_times),
        "completion_time_max": max(completion_times, default=0),
        "completion_time_min": min(completion_times, default=0),
        "total_time_avg": compute_average(total_times),
        "total_time_max": max(total_times, default=0),
        "total_time_min": min(total_times, default=0),
        "ttft_avg": compute_average(ttft_times),
        "ttft_max": max(ttft_times, default=0),
        "ttft_min": min(ttft_times, default=0),
        "tpot_avg": compute_average(tpot_times),
        "tpot_max": max(tpot_times, default=0),
        "tpot_min": min(tpot_times, default=0),
        "prompt_tokens_avg": compute_average(prompt_tokens),
        "prompt_tokens_max": max(prompt_tokens, default=0),
        "prompt_tokens_min": min(prompt_tokens, default=0),
        "completion_tokens_avg": compute_average(completion_tokens),
        "completion_tokens_max": max(completion_tokens, default=0),
        "completion_tokens_min": min(completion_tokens, default=0),
    }

    return metrics

def main(input_file):
    with open(input_file, "r") as f:
        results = [json.loads(line) for line in f if line.startswith('{')]
    
    metrics = compute_metrics(results)

    print("Metrics (times in seconds):")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics from evaluation results.")
    parser.add_argument("--input_file", "-i", type=str, required=True, help="Path to the input JSONL file with evaluation results.")
    
    args = parser.parse_args()
    main(args.input_file)

