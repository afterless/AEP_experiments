import argparse
import json
import os
import gc
from collections import defaultdict
from pathlib import Path
from itertools import product

import torch
import submitit
from submitit.helpers import Checkpointable
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def load_steer_vector(model_name: str, vec_slice: slice, behavior: str, vec_pth: str):
    behavior_files = f"{model_name}-{behavior}-*.json"
    files = list(Path(vec_pth).glob(behavior_files))[vec_slice]
    for file in files:
        with open(file, "r") as f:
            vector = torch.tensor(json.load(f))
        yield vector 

def add_special_tokens(item, ab_mode):
    prompt = item["question"]
    if ab_mode:
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}\n\nOnly output your final answer, A or B.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
    else:
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}\n\nOnly output your final answer, Yes or No.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
    return {"prompt": formatted_prompt}

def apply_steer_vector(vector, strength):
    def hook(model, inp, out):
        # Move vector to same device as model outputs and add scaled vector to last token's output.
        v = vector.to(out[0].device)
        out[0][:, -1] += strength * v
        return out
    return hook

def evaluate_model(model, ds, tokenizer, ab_mode):
    ds_size = len(ds)
    # For each example, select target from dataset's field "answer_matching_behavior"
    if ab_mode:
        targets = [s[2] for s in ds["answer_matching_behavior"]]
    else:
        targets = [s[1:] for s in ds["answer_matching_behavior"]]

    ds = ds.map(lambda item: add_special_tokens(item, ab_mode), num_proc=16)
    input_ids = tokenizer(ds["prompt"], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    prompt_len = len(input_ids["input_ids"][0])
    with torch.inference_mode():
        gen_ids = model.generate(
            **input_ids, 
            max_new_tokens=5, 
            do_sample=False, 
            temperature=None,
            top_p=None,
            use_cache=True, 
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = gen_ids[..., prompt_len:]
    gen_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    score = sum(1 for ans, tgt in zip(gen_text, targets) if tgt in ans) / ds_size * 100

    del input_ids, gen_ids, gen_text
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return score

class IDEvaluator(Checkpointable):
    def __init__(self, model_name: str, output_dir: str, vector_path: str,
                 behavior: str, strengths: list[int], vec_slice: slice):

        self.model_name = model_name
        self.dir = output_dir
        self.vector_path = vector_path
        self.behavior = behavior

        self.strengths = strengths
        self.curr_strengths = list(strengths)

        self.vec_slice = vec_slice
        self.results = {}

    def __call__(self):
        # Load model and tokenizer based on model_name.
        if self.model_name == "Llama-3.1-405B-Instruct":
            model = AutoModelForCausalLM.from_pretrained(
                "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
            tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                f"meta-llama/Meta-{self.model_name}",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-{self.model_name}")

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        # Set dataset and evaluation mode based on behavior.
        ab_mode = False
        if self.behavior in ["power-seeking-inclination", "self-awareness-general-ai", "corrigible-neutral-HHH"]:
            benchmark_ds = load_dataset("Anthropic/model-written-evals", data_dir="advanced-ai-risk/lm_generated_evals",
                                        data_files=f"{self.behavior}.jsonl", split='train[-20%:]')
            ab_mode = True
        elif self.behavior == "openness":
            benchmark_ds = load_dataset("json", data_files=f"{self.behavior}.jsonl", split='train[-20%:]')
        else:
            benchmark_ds = load_dataset("Anthropic/model-written-evals", data_dir="persona",
                                        data_files=f"{self.behavior}.jsonl", split='train[-20%:]')

        if self.behavior not in self.results:
            self.results[self.behavior] = defaultdict(list)
            baseline = evaluate_model(model, benchmark_ds, tokenizer, ab_mode)
            self.results[self.behavior]["baseline"].append(baseline)

        layer_idx = 7 if self.model_name.split("-")[2] == "8B" else 66

        # For each strength that hasn't been processed, compute the steering differences.
        for s in self.curr_strengths.copy():
            for vec in load_steer_vector(self.model_name, self.vec_slice, self.behavior, self.vector_path):
                handle = model.model.layers[layer_idx].register_forward_hook(
                    apply_steer_vector(vec[layer_idx], s)
                )
                steer_score = evaluate_model(model, benchmark_ds, tokenizer, ab_mode)
                handle.remove()
                self.results[self.behavior][s].append(steer_score - self.results[self.behavior]["baseline"][0])
            self.curr_strengths.remove(s)

        # Save the results to file.
        os.makedirs(self.dir, exist_ok=True)
        out_filepath = os.path.join(
            self.dir,
            f"strengths-{self.model_name}-{self.behavior}-{self.strengths[0]}_{self.strengths[-1]}-{self.vec_slice.start}_{self.vec_slice.stop-1}.json" 
        )
        with open(out_filepath, "w") as f:
            json.dump(self.results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True,
                        choices=["Llama-3-8B-Instruct", "Llama-3-70B-Instruct", "Llama-3.1-405B-Instruct"],
                        help="Model to evaluate")
    parser.add_argument("-d", "--dir", type=str, default="ID_eval_results",
                        help="Directory to store results in")
    parser.add_argument("-v", "--vector_path", type=str, default="fib_sv",
                        help="Directory where vectors are stored")
    args = parser.parse_args()

    # Define the combinations to evaluate.
    behaviors = [
        "agreeableness",
        "conscientiousness",
        "corrigible-neutral-HHH",
        "extraversion",
        "neuroticism",
        "openness",
        "politically-liberal",
        "power-seeking-inclination",
        "self-awareness-general-ai"
    ]

    vector_slices = [slice(i, i+2) for i in range(0, 10, 2)]
    strengths = [[i, i+2] for i in range(-10, 9, 4)] + [[10]]

    # Set up the submitit executor.
    executor = submitit.AutoExecutor(folder="logs", slurm_max_num_timeout=3)
    executor.update_parameters(timeout_min=120, slurm_gres="gpu:H200:1", cpus_per_task=32, slurm_mem_per_cpu="4G")
    
    # Submit one job per combination of behavior, vector slice, and steering strength.
    with executor.batch():
        for behavior, vec_slice, strength in product(behaviors, vector_slices, strengths):
            evaluator = IDEvaluator(args.model, args.dir, args.vector_path, behavior, strength, vec_slice)
            _ = executor.submit(evaluator)