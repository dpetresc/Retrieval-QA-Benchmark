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
    ttft_key: str = "model.local.ttft.profile"
    tpot_key: str = "model.local.tpot.profile"

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

        past_key_values = None
        generated_tokens = []
        next_token_id = inputs["input_ids"]
        
        # Measure TTFT
        torch.cuda.synchronize()
        start_ttft = time.time()
        next_logits, past_key_values = self.model(next_token_id, past_key_values=past_key_values, use_cache=True).to_tuple()
        ttft = (time.time() - start_ttft) * 1000  # TTFT in ms

        # Append the first token
        next_token_id = torch.argmax(next_logits[:, -1:], dim=-1)
        generated_tokens.append(next_token_id.item())

        # Measure TPOT for subsequent tokens
        torch.cuda.synchronize()
        start_tpot = time.time()
        for _ in range(self.run_args.get("max_new_tokens", 50) - 1):
            next_logits, past_key_values = self.model(next_token_id, past_key_values=past_key_values, use_cache=True).to_tuple()
            next_token_id = torch.argmax(next_logits[:, -1:], dim=-1)
            generated_tokens.append(next_token_id.item())

            # Stop if EOS token is generated
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        total_time = (time.time() - start_tpot) * 1000  # Total generation time in ms

        # Calculate TPOT
        total_tokens = len(generated_tokens)
        tpot = total_time / (total_tokens - 1) if total_tokens > 1 else 0

        # Decode generated text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Update profiling metrics
        if self.completion_time_key not in PROFILER.accumulator:
            PROFILER.accumulator[self.completion_time_key] = 0
            PROFILER.counter[self.completion_time_key] = 0

        if self.ttft_key not in PROFILER.accumulator:
            PROFILER.accumulator[self.ttft_key] = 0
            PROFILER.counter[self.ttft_key] = 0

        if self.tpot_key not in PROFILER.accumulator:
            PROFILER.accumulator[self.tpot_key] = 0
            PROFILER.counter[self.tpot_key] = 0

        PROFILER.accumulator[self.completion_time_key] += total_time
        PROFILER.counter[self.completion_time_key] += 1

        PROFILER.accumulator[self.ttft_key] += ttft
        PROFILER.counter[self.ttft_key] += 1

        PROFILER.accumulator[self.tpot_key] += tpot
        PROFILER.counter[self.tpot_key] += 1

        return BaseLLMOutput(
            generated=generated_text,
            completion_tokens=total_tokens,
            prompt_tokens=inputs["input_ids"].shape[1],
            full_output=generated_text
        )


#    def generate(self, text: str) -> BaseLLMOutput:
#        input_text = "\n".join([self.system_prompt, text])
#        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
#        t0 = time.time()
#
#        output = self.model.generate(
#            inputs["input_ids"],
#            #max_length=self.run_args.get("max_new_tokens", 50),
#            num_return_sequences=1,
#            pad_token_id=self.tokenizer.eos_token_id,
#            **self.run_args,
#        )
#
#        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
#        # Extract the generated part of the text
#        generated_text_ = generated_text[len(input_text):].strip()
#        #print(generated_text)
#        t_gen = (time.time() - t0) * 1000  # ms
#
#         # Initialize profiler keys if not already present
#        if self.completion_time_key not in PROFILER.accumulator:
#            PROFILER.accumulator[self.completion_time_key] = 0
#            PROFILER.counter[self.completion_time_key] = 0
#
#        if self.ttft_key not in PROFILER.accumulator:
#            PROFILER.accumulator[self.ttft_key] = 0
#            PROFILER.counter[self.ttft_key] = 1
#
#        if self.tpot_key not in PROFILER.accumulator:
#            PROFILER.accumulator[self.tpot_key] = 0
#            PROFILER.counter[self.tpot_key] = 1
#
#        PROFILER.accumulator[self.completion_time_key] += t_gen
#        PROFILER.counter[self.completion_time_key] += 1
#
#        #print(PROFILER.accumulator, PROFILER.counter)
#
#        return BaseLLMOutput(
#            generated=generated_text_,
#            completion_tokens=output.shape[1] - inputs["input_ids"].shape[1],
#            prompt_tokens=inputs["input_ids"].shape[1],
#            full_output=generated_text
#        )
