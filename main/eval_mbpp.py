import os
import sys
from tqdm import tqdm
import json
import math

import fire
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def main(
        output_dir: str = "./output_dir",
        base_model: str = "huggyllama/llama-7b",
        prompt_template: str = "",
        batch_size: int = 64,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_return_sequences: int = 1,
):
    os.makedirs(output_dir, exist_ok=True)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world_size != 1

    if ddp:
        # init distributed process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
    )

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    tokenizer.padding_size = "left"

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    print("Enter your prompt (type 'exit' to stop):")
    while True:
        prompt = input("> ")
        if prompt.lower() == "exit":
            break

        prompts = [prompt]

        input_ids = tokenizer(prompts, return_tensors="pt", truncation=True, padding="longest").input_ids.to(device)

        try:
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                num_return_sequences=num_return_sequences,
            )

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    max_new_tokens=128,
                )
        except torch.cuda.OutOfMemoryError:
            print("Out of memory ... continue ...")
            torch.cuda.empty_cache()
            continue

        output_strings = [tokenizer.decode(s, skip_special_tokens=True, ignore_tokenization_space=True) for s in output_ids]

        print("Generated Code:")
        for generated_code in output_strings:
            print(generated_code)

if __name__ == "__main__":
    fire.Fire(main)
