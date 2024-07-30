import re
import time
from typing import Any, Dict, Optional, Union

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizerFast
import torch

from retrieval_qa_benchmark.schema.model import BaseLLM, BaseLLMOutput
from retrieval_qa_benchmark.utils.profiler import PROFILER
from retrieval_qa_benchmark.utils.registry import REGISTRY


@REGISTRY.register_model("local-llm")
class LocalLLM(BaseLLM):
    model: Union[AutoModelForCausalLM, LlamaForCausalLM]
    tokenizer: Union[AutoTokenizer, LlamaTokenizerFast]
    device: torch.device
    system_prompt: str = "You are a helpful assistant."
    #boot_time_key: str = "model.local.boot.profile"
    completion_time_key: str = "model.local.completion.profile"

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def build(
        cls,
        name: str = "llama2-13b-chat-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        system_prompt: Optional[str] = None,
        run_args: Optional[Dict[str, Any]] = None,
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ) -> "LocalLLM":
        model = AutoModelForCausalLM.from_pretrained(name, token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(name, token=hf_token)
        device = torch.device(device)
        model.to(device)
        return cls(
            name=name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            run_args=run_args or {},
            system_prompt=system_prompt or "",
            **kwargs,
        )

    def generate(self, text: str) -> BaseLLMOutput:
        input_text = "\n".join([self.system_prompt, text])
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        t0 = time.time()

        output = self.model.generate(
            inputs["input_ids"],
            #max_length=self.run_args.get("max_new_tokens", 50),
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            **self.run_args,
        )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Extract the generated part of the text
        generated_text_ = generated_text[len(input_text):].strip()
        #print(generated_text)
        t_gen = (time.time() - t0) * 1000  # ms

         # Initialize profiler keys if not already present
        if self.completion_time_key not in PROFILER.accumulator:
            PROFILER.accumulator[self.completion_time_key] = 0
            PROFILER.counter[self.completion_time_key] = 0

        PROFILER.accumulator[self.completion_time_key] += t_gen
        PROFILER.counter[self.completion_time_key] += 1

        #print(PROFILER.accumulator, PROFILER.counter)

        return BaseLLMOutput(
            generated=generated_text_,
            completion_tokens=output.shape[1] - inputs["input_ids"].shape[1],
            prompt_tokens=inputs["input_ids"].shape[1],
            full_output=generated_text
        )

