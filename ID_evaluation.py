import argparse
from collections import defaultdict
from pathlib import Path
import gc
import os
import json
from functools import partial

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def load_steer_vector(model_name: str, ranges: slice, behavior: str, vec_pth: str):
    behavior_files = f"{model_name}-{behavior}-*.json"
    files = list(Path(vec_pth).glob(behavior_files))[ranges]
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
        v = vector.to(out[0].device)
        out[0][:, -1] += strength * v
        return out
    return hook

def evaluate_model(model, ds, tokenizer, ab_mode):
    ds_size = len(ds)
    
    if ab_mode:
        targets = [s[2] for s in ds["answer_matching_behavior"]]
    else:
        targets = [s[1:] for s in ds["answer_matching_behavior"]]

    ds = ds.map(partial(add_special_tokens, ab_mode=ab_mode), num_proc=16)
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
    score = sum([1 for ans, tgt in zip(gen_text, targets) if (tgt in ans)]) / ds_size * 100

    del input_ids, gen_ids, gen_text
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return score

def parse_range(arg):
    try:
        arg = arg.strip("[]")
        start, end = map(int, arg.split(","))
        return slice(start, end+1)
    except ValueError:
        raise argparse.ArgumentTypeError("Range must be in the format [start, end]")

def parse_behaviors(arg):
    choices = {
        "agreeableness",
        "conscientiousness", 
        "corrigible-neutral-HHH", 
        "extraversion",
        "neuroticism",
        "openness",
        "politically-liberal",
        "power-seeking-inclination",
        "self-awareness-general-ai"
    }
    behaviors = [b.strip() for b in arg.split(",")]
    invalid = [b for b in behaviors if b not in choices]
    if invalid:
        raise argparse.ArgumentTypeError(f'Invalid behaviors: {", ".join(invalid)}')
    return behaviors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, choices=["Llama-3-8B-Instruct", "Llama-3-70B-Instruct", "Llama-3.1-405B-Instruct"], help="model to plot for")
    parser.add_argument("-b", "--behaviors", type=parse_behaviors, required=True, help="behaviors to select for")
    parser.add_argument("-s", "--strengths", type=parse_range, required=True, help="Stength Range in the format [start,end]")
    parser.add_argument("-r", "--ranges", type=parse_range, required=True, help="Slice Range in the format [start,end]")
    parser.add_argument("-d", "--dir", type=str, default="ID_eval_results", help="directory to store results in")
    parser.add_argument("-v", "--vector_path", type=str, default="fib_sv", help="directory where vectors are stored")
    args = parser.parse_args()

    if args.model == "Llama-3.1-405B-Instruct":
        model = AutoModelForCausalLM.from_pretrained("unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit")
    else:
        model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-{args.model}", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-{args.model}")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    results = {}
    for b in args.behaviors:
        results[b] = defaultdict(list)

        ab_mode = False
        if b in ["power-seeking-inclination", "self-awareness-general-ai", "corrigible-neutral-HHH"]:
            benchmark_ds = load_dataset("Anthropic/model-written-evals", data_dir="advanced-ai-risk/lm_generated_evals", data_files=f"{b}.jsonl", split='train[-20%:]')
            ab_mode = True
        elif b == "openness":
            benchmark_ds = load_dataset("json", data_files=f"{b}.jsonl", split='train[-20%:]')
        else:
            benchmark_ds = load_dataset("Anthropic/model-written-evals", data_dir="persona", data_files=f"{b}.jsonl", split='train[-20%:]')

        baseline = evaluate_model(model, benchmark_ds, tokenizer, ab_mode)
        results[b]["baseline"].append(baseline)
        layer_idx = 7 if args.model.split("-")[2] == "8B" else 29
        for s in range(args.strengths.start, args.strengths.stop+1, 2):
            for vec in load_steer_vector(args.model, args.ranges, b, args.vector_path):
                handle = model.model.layers[layer_idx].register_forward_hook(apply_steer_vector(vec[layer_idx], s))
                steer_score = evaluate_model(model, benchmark_ds, tokenizer, ab_mode)
                handle.remove()
                results[b][s].append(steer_score - results[b]["baseline"][0])
                
    os.makedirs(args.dir, exist_ok=True)
    out_filepath = os.path.join(
        args.dir,
        f"strengths-{args.model}-{'_'.join(args.behaviors)}-{args.strengths.start}_{args.strengths.stop-1}-{args.ranges.start}_{args.ranges.stop-1}.json" 
    )
    with open(out_filepath, "w") as f:
        json.dump(results, f, indent=4)