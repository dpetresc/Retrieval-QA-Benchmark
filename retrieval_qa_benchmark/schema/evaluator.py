from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Extra
from tqdm import tqdm

from retrieval_qa_benchmark.schema.dataset import BaseDataset
from retrieval_qa_benchmark.schema.datatypes import (
    QAPrediction,
    QARecord,
)
from retrieval_qa_benchmark.schema.model import BaseLLM
from retrieval_qa_benchmark.schema.transform import TransformGraph
from retrieval_qa_benchmark.utils.profiler import PROFILER


def default_matcher(x: str, y: QARecord) -> float:
    return float(x == y.answer)

def load_sampled_questions(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    questions = [line.split("Question: ")[1].strip() for line in lines if line.startswith("Question: ")]
    return questions

class BaseEvaluator(BaseModel):
    """Base class for evaluators"""

    dataset: BaseDataset
    llm: BaseLLM
    transform: TransformGraph
    matcher: Callable[[str, QARecord], float] = default_matcher
    out_file: Optional[str] = None
    sampled_questions_file: Optional[str] = None

    class Config:
        extra = Extra.forbid

    def filter_sampled_questions(self, eval_set: List[QARecord]) -> List[QARecord]:
        sampled_questions = load_sampled_questions(self.sampled_questions_file)
        return [record for record in eval_set if record.question in sampled_questions]

    def __call__(self) -> Tuple[float, List[QAPrediction]]:
        """Main evaluator pipeline

        If the transform returns :class:`QAPrediction`
        then LLM will not be called to generate answers.

        :raises e: _description_
        :return: _description_
        :rtype: Tuple[float, List[QAPrediction]]
        """
        PROFILER.clear()
        result: List[QAPrediction] = []
        
        self.sampled_questions_file = "/mnt/nfs/home/dpetresc/Retrieval-QA-Benchmark/scripts/sampled_questions.txt"
        eval_set = self.dataset.eval_set
        if self.sampled_questions_file:
            eval_set = self.filter_sampled_questions(eval_set)


        for d in tqdm(eval_set, desc="Evaluating"):
            try:
                d_ = self.transform(d)
                if type(d_) is QARecord:
                    pred = self.llm(d_)
                    prompt_tokens = pred.prompt_tokens
                    completion_tokens = pred.completion_tokens
                    if d_.stack and len(d_.stack) > 0:
                        prompt_tokens += sum([p.prompt_tokens for p in d_.stack])
                        completion_tokens += sum(
                            [p.completion_tokens for p in d_.stack]
                        )
                    mtch = self.matcher(pred.generated, d_)
                    profile_avg = {
                        k: PROFILER.accumulator[k] / PROFILER.counter[k]
                        for k in PROFILER.accumulator.keys()
                    }
                    result.append(
                        QAPrediction(
                            **d.model_dump(),
                            generated=pred.generated,
                            matched=mtch,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            profile_avg=profile_avg,
                            profile_count=PROFILER.counter,
                            profile_time=PROFILER.accumulator,
                            full_output=pred.full_output
                        )
                    )
                elif type(d_) is QAPrediction:
                    pred = d_  # type: ignore
                    pred.matched = self.matcher(pred.generated, d_)
                    pred.profile_count = PROFILER.counter
                    pred.profile_time = PROFILER.accumulator
                    pred.profile_avg = {
                        k: PROFILER.accumulator[k] / PROFILER.counter[k]
                        for k in PROFILER.accumulator.keys()
                    }
                    result.append(pred)  # type: ignore
                else:
                    raise TypeError(f"Unknown returned type `{type(d_)}`")
                PROFILER.clear()
            except Exception as e:
                logger.error(f"Failed to evaluate record {str(d)}")
                raise e
        score = 100 * sum([p.matched for p in result]) / len(self.dataset)
        logger.info(
            f"Evaluation finished! Executed Evaluator:{type(self)} on "
            f"Dataset:{self.dataset.name} with Model:{self.llm.name}. "
            f"Score: {score:.2f}%"
        )
        if self.out_file:
            with open(self.out_file, "w") as f:
                f.write("\n".join([r.model_dump_json() for r in result]))
        return score, result
