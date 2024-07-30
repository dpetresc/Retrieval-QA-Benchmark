import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class LocalLLM:
    def __init__(self, name, device="cuda" if torch.cuda.is_available() else "cpu", hf_token=None):
        self.model = AutoModelForCausalLM.from_pretrained(name, token=hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained(name, token=hf_token)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.system_prompt = ""
        self.run_args = {"max_new_tokens": 50}
        self.completion_time_key = "model.local.completion.profile"

    def generate(self, text):
        input_text = "\n".join([self.system_prompt, text])
        print("INPUT TEXT ", input_text)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        print("INPUTS ", inputs)
        t0 = time.time()
        
        print(self.run_args)
        output = self.model.generate(
            inputs["input_ids"],
            #max_length=inputs["input_ids"].shape[1] + self.run_args.get("max_new_tokens", 50),
            #max_new_tokens=self.run_args.get("max_new_tokens", 50),
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            **self.run_args,
        )

        print("OUTPUT ", output)

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        t_gen = (time.time() - t0) * 1000  # ms
        print("Generated text without clean ", generated_text)

        # Extract the generated part of the text
        generated_text = generated_text[len(input_text):].strip()

        print(f"Generated text: {generated_text}")
        print(f"Time taken: {t_gen} ms")

        return {
            "generated": generated_text,
            "completion_tokens": output.shape[1] - inputs["input_ids"].shape[1],
            "prompt_tokens": inputs["input_ids"].shape[1],
        }

if __name__ == "__main__":
    # Set your Hugging Face token
    hf_token = "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"

    # Initialize the model
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    local_llm = LocalLLM(name=model_name, hf_token=hf_token)

    # Test input
    test_input = "[INST]The following are multiple choice questions (with answers) with context:\n\nQuestion: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\tB. 4\tC. 2\tD. 6\n[/INST]Answer: \n\n"

    # Generate the output
    output = local_llm.generate(test_input)

    # Print the results
    print(f"Prompt tokens: {output['prompt_tokens']}")
    print(f"Completion tokens: {output['completion_tokens']}")
    print(f"Generated text: {output['generated']}")

