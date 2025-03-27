import os
import gc
import argparse
import json

from pathlib import Path
from typing import Literal
from tqdm import tqdm
from itertools import product
from collections import defaultdict

import torch
import submitit
from submitit.helpers import Checkpointable
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def load_steer_vector(model_name: str, ranges: slice, behavior: str, vec_pth: str):
    behavior_files = f"{model_name}-{behavior}-*.json"
    files = list(Path(vec_pth).glob(behavior_files))[ranges]
    for file in files:
        with open(file, "r") as f:
            vector = torch.tensor(json.load(f))
        yield vector 

def add_special_tokens(example):
    answer_map = ["(A\nA", "(B\nB", "(C\nC", "(D\nD"]
    formatted_prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        f'{example["question"]}\n(A) {example["choices"][0]}\n(B) {example["choices"][1]}\n(C) {example["choices"][2]}\n(D) {example["choices"][3]}\n\n'
        f"Think in step by step detail, and end your response with the final answer as either (A), (B), (C), or (D), nothing else after."
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
    return {"answer_str": answer_map[example["answer"]], "prompt": formatted_prompt}

def apply_steer_vector(vector, strength):
    def hook(model, inp, out):
        v = vector.to(out[0].device)
        out[0][:, -1] += strength * v
        return out
    return hook

def evaluate_model(model, ds, tokenizer, bs=128, max_new_tokens=1024):
    ds = ds.map(add_special_tokens, num_proc=16)
    score = 0
    total_batches = 0
    for i in tqdm(range(0, len(ds), bs)):
        batch = ds.select(range(i, i + bs)) if i + bs < len(ds) else ds.select(range(i, len(ds)))
        input_ids = tokenizer(batch["prompt"], return_tensors="pt", padding=True, pad_to_multiple_of=64, add_special_tokens=False).to(model.device)
        targets = tokenizer(batch["answer_str"], return_tensors="pt", add_special_tokens=False).to(model.device)
        targets = torch.masked_select(targets["input_ids"], targets["input_ids"] != 198).view(-1, 2)
        with torch.inference_mode():
            gen_ids = model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        answer_indices = torch.where(torch.isin(gen_ids, torch.tensor([32, 33, 34, 35, 4444, 5462, 3100, 5549], device=model.device)), torch.arange(gen_ids.shape[1], device=model.device), -1).max(dim=-1).values
        answer_tokens = gen_ids[torch.arange(gen_ids.shape[0]), answer_indices]
        score += (answer_tokens.unsqueeze(1) == targets).any(dim=1).sum() / len(batch) * 100
        total_batches += 1
        del input_ids, gen_ids, answer_indices, answer_tokens
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return score.item() / total_batches

class NetworkEvaluator(Checkpointable):
    def __init__(self,
                model_name: str, 
                output_dir: str, 
                vector_pth: str, 
                behaviors: list[str],
                ranges: slice,
                strengths: list[int]
            ):
        self.model_name = model_name
        self.dir = output_dir
        self.vector_pth = vector_pth

        self.curr_behaviors = list(behaviors)
        self.behaviors = behaviors
        self.curr_strengths = list(strengths)
        self.strengths = strengths

        self.ranges = ranges

        self.results = {}
    
    def __call__(self):
        if self.model_name == "Llama-3.1-405B-Instruct":
            model = AutoModelForCausalLM.from_pretrained("unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit", device_map="auto", attn_implementation="flash_attention_2")
            tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit")
        else:
            model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-{self.model_name}", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-{self.model_name}")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        layer_idx = 7 if self.model_name.split("-")[2] == "8B" else 66 
        ds = load_dataset("cais/mmlu", "all", split="dev")

        if "baseline" not in self.results:
            self.results["baseline"] = evaluate_model(model, ds, tokenizer)

        for b in self.curr_behaviors.copy():
            self.results[b] = defaultdict(list) if b not in self.results else self.results[b]
            for s in self.curr_strengths.copy():
                for vec in load_steer_vector(self.model_name, self.ranges, b, self.vector_pth):
                    handle = model.model.layers[layer_idx].register_forward_hook(apply_steer_vector(vec[layer_idx], s))
                    steer_score = evaluate_model(model, ds, tokenizer)
                    handle.remove()
                    self.results[b][s].append(steer_score - self.results["baseline"])
                self.curr_strengths.remove(s)
            self.curr_behaviors.remove(b)
            
        os.makedirs(self.dir, exist_ok=True)
        out_filepath = os.path.join(
            self.dir,
            f"{self.model_name}-{'_'.join(self.behaviors)}-{self.strengths[0]}_{self.strengths[-1]}-{self.ranges.start}_{self.ranges.stop-1}.json" 
        )
        with open(out_filepath, "w") as f:
            json.dump(self.results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, choices=["Llama-3-8B-Instruct", "Llama-3-70B-Instruct", "Llama-3.1-405B-Instruct"], help="model to plot for")
    parser.add_argument("-d", "--dir", type=str, default="MMLU_eval_results", help="directory to store results in")
    parser.add_argument("-v", "--vector_path", type=str, default="fib_sv", help="directory where vectors are stored")
    args = parser.parse_args()

    behaviors=[
        ["agreeableness"],
        ["conscientiousness"],
        ["corrigible-neutral-HHH"],
        ["extraversion"],
        ["neuroticism"],
        ["openness"],
        ["politically-liberal"],
        ["power-seeking-inclination"],
        ["self-awareness-general-ai"]
    ]
    ranges = [slice(i, i+2) for i in range(0, 10, 2)]
    strengths = [[i, i+2] for i in range(-10, 9, 4)] + [[10]]

    executor = submitit.AutoExecutor(folder="logs", slurm_max_num_timeout=3)
    executor.update_parameters(timeout_min=120, slurm_gres="gpu:H200:2", cpus_per_task=32, slurm_mem_per_cpu="4G")
    with executor.batch():
        for b, r, s in product(behaviors, ranges, strengths):
            evaluator = NetworkEvaluator(args.model, args.dir, args.vector_path, b, r, s)
            _ = executor.submit(evaluator)

