from pathlib import Path
import os
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# SETTINGS - START
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
SV_DIR = "train_set_sv"  # directory containing steering vectors
# BEHAVIORS = ["desire-to-persuade-people-to-be-more-honest-to-others", "agreeableness", "conscientiousness", "extraversion", "openness", "neuroticism", "politically-liberal"]  # example behavior for benchmark and steering vectors file
BEHAVIORS = ["power-seeking-inclination", "self-awareness-general-ai", "corrigible-neutral-HHH"]  # example behavior for benchmark and steering vectors file
STEERING_VECTORS = [2, 4, 6, 8, 10] # contains stop decile index, assuming starting from 1 i.e. 1: 0-0.1, 6: 0-0.6
STEERING_STRENGTHS = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
MAX_NEW_TOKENS = 700
FORMAT_PROMPT = True #dont do false lol, model will be in the wrong 'mode' and think before answering or do other stuff
NUM_PROC = 16
OUT_DIR = "MMLU_eval_results" # output directory
EVAL_TARGET = 'strengths'

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
TAG = "CAA" #for output file naming

# SETTINGS - END

# Define a hook generator to inject a steering vector at the given layer
def get_hook(layer_idx, steer_vector, strength):
    def hook(module, inputs, output):
        # steer_vector is of shape (hidden_size,)
        v = steer_vector.to(output[0].device)
        # Modify the hidden state of the LAST token only.
        output[0][:, -1] += strength*v
        return output
    return hook

def add_special_tokens(item):
    prompt = item["question"]
    choices = item["choices"]
    # Apply transformation to the prompt text.
    formatted_prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}\n\n{choices}\n\n"
        f"Please describe your resoning in thorough detail, and afterwards output your final answer as either 0, 1, 2, or 3 only, nothing else."
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
    return {"prompt": formatted_prompt}


# Evaluation function: for a (small) benchmark dataset compute %
# of generated answers that contain the answer_matching_behavior string.
def evaluate_model(model, ds, tokenizer):
    ds = ds.map(add_special_tokens, num_proc=NUM_PROC)
    score = 0
    for i in range(0, len(ds), 12):
        mini_ds = ds.select(range(i, i+12))
        ds_size = len(mini_ds)
        targets = [s for s in mini_ds["answer"]]
        input_ids = tokenizer(mini_ds["prompt"], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

        with torch.inference_mode():
            gen_ids = model.generate(
                **input_ids, 
                max_new_tokens=MAX_NEW_TOKENS, 
                do_sample=False, 
                temperature=None,
                top_p=None,
                use_cache=True, 
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            )
        new_tokens = gen_ids[..., :]
        gen_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        import pdb; pdb.set_trace()
        score += sum([1 for ans, tgt in zip(gen_text, targets) if (tgt in ans[-1])]) / ds_size * 100
    return score

def main():
    for behavior in BEHAVIORS:
        # Load benchmark dataset (using the JSONL file from ./evals/)
        benchmark_ds = load_dataset("cais/mmlu", "all", split="validation")
        baseline = evaluate_model(model, benchmark_ds, tokenizer)
        results = {}
        results["baseline"] = baseline
        for sv in STEERING_VECTORS:
            # e.g., "Llama-3-8B-Instruct-corrigible-neutral-HHH-train.json"
            behavior_files = f"Llama-3-8B-Instruct-{behavior}-*-*.json"
            files = Path(f"./{SV_DIR}").glob(behavior_files)
            file_list = list(files)[:sv]
            steering_tensor = None
            for file in file_list:
                with open(file, "r") as f:
                    vector = torch.tensor(json.load(f))
                    if steering_tensor is None:
                        steering_tensor = vector
                    else:
                        steering_tensor.add_(vector)
            steering_tensor /= len(file_list)
            result_sv = {}
            for strength in STEERING_STRENGTHS:                
                layer_scores = []
                layer_indices = []
                num_layers = steering_tensor.shape[0]
                
                # Evaluate model for each steering layer modification
                for i in range(0, num_layers, 2):
                    # Register hook at corresponding layer.
                    # Note: steering_tensor was computed from layers[1:], so real layer index is (i+1).
                    layer_idx = i + 1  
                    hook = get_hook(layer_idx, steering_tensor[i], strength)
                    handle = model.model.layers[layer_idx].register_forward_hook(hook)
                    
                    print(f"Evaluating steering at layer {layer_idx}...")
                    steer_score = evaluate_model(model, benchmark_ds, tokenizer)
                    print(f"Layer {layer_idx}: Answer Matching Behavior % = {steer_score:.2f}")

                    handle.remove()  # remove hook after evaluation
                    
                    layer_indices.append(layer_idx)
                    layer_scores.append(steer_score - baseline)
                    
                result_sv[strength] = torch.tensor(layer_scores).mean().item()
                
            # Plot results
            if sv == 10:
                plt.figure(figsize=(8, 5))
                plt.plot(STEERING_STRENGTHS, list(result_sv.values()), marker='o', color='blue')
                plt.xlabel("Strength Applied")
                plt.xticks(STEERING_STRENGTHS)
                plt.ylabel("Change in Matching Behavior (%)")
                plt.title("Change in MMLU Accuracy via Steering Strength")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"strength-vs-{behavior}-{TAG}.png")
                plt.show()
                
            results[sv] = result_sv

        os.makedirs(OUT_DIR, exist_ok=True)
        out_filepath = os.path.join(OUT_DIR, f"{EVAL_TARGET}-Llama-3-8B-Instruct-{TAG}-{behavior}.json")
        with open(out_filepath, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()