#!/usr/bin/python3

from os import path, makedirs
from argparse import ArgumentParser
from loguru import logger
from retrieval_qa_benchmark.datasets import *
from retrieval_qa_benchmark.models import *
from retrieval_qa_benchmark.evaluators import *
from retrieval_qa_benchmark.transforms import *
from retrieval_qa_benchmark.utils.config import load
from retrieval_qa_benchmark.utils.factory import EvaluatorFactory

def get_model_name(config):
    model_config = config.get("evaluator", {}).get("model", {})
    model_name = model_config.get("name", "unknown_model")
    return model_name

p = ArgumentParser("Evaluation script for MMLU dataset")
p.add_argument("--config", "-c", default="../config/mmlu.yaml")
p.add_argument("--mmlu-subset", "-set", default="prehistory")
#p.add_argument("--mmlu-subset", "-set", default="all")
p.add_argument("--outdir", "-o", default="results")
p.add_argument("--topk", "-k", default=5)

args = p.parse_args()
config = load(open(args.config))
if "args" in config["evaluator"]["dataset"]:
    config["evaluator"]["dataset"]["args"] = {}
assert (
    config["evaluator"]["dataset"]["type"] == "mmlu"
), "This script is only for MMLU dataset"

config["evaluator"]["dataset"]["args"] = {"subset": args.mmlu_subset}
logger.info(f"Evaluating MMLU-{config['evaluator']['dataset']['args']['subset']}")

config_name = path.basename(args.config).replace('.yaml', '')
model_name = get_model_name(config)

outfile_result = path.join(
    args.outdir, f"mmlu_{args.mmlu_subset}", f"{args.topk}_{config_name}_{model_name}.jsonl"
)
logger.info(f"output_file: {outfile_result}")

evaluator: MCSAEvaluator = EvaluatorFactory.from_config(config).build()
acc, matched = evaluator()

avg_token = sum([m.prompt_tokens + m.completion_tokens for m in matched]) / len(matched)

makedirs(path.join(args.outdir, f"mmlu_{args.mmlu_subset}"), exist_ok=True)

with open(outfile_result, "w", encoding="utf-8") as f:
    f.write(
        "\n".join(
            [f"Accuracy: {acc:.2f}%", f"Average tokens: {avg_token:.2f}"]
            + [r.model_dump_json() for r in matched]
        )
    )
