from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import gc
from datasets import load_dataset
from more_itertools import chunked
from tqdm import tqdm
from torch.utils.data import DataLoader
import pdb

### SETTINGS - START

#os.environ['HF_HOME'] = '/tmp' #remember to request tmp memory during salloc

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name_short = "Llama-3-8B-Instruct"	
data_dir = "advanced-ai-risk/lm_generated_evals"
behaviors = [
    "power-seeking-inclination", 
    "self-awareness-general-ai", 
    "corrigible-neutral-HHH"
]
# data_dir = "persona"
# behaviors = [
#     "conscientiousness",
#     "extraversion",
#     "agreeableness",
#     "neuroticism",
#     "openness",
#     "politically-liberal"
# ]
OUT_DIR = "caa_sv"

deciles = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_layers = len(model.model.layers)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

### SETTINGS - END

def add_special_tokens(questions, pos_answers, neg_answers):
    pos_prompts = [f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{a}" for q, a in zip(questions, pos_answers)]
    neg_prompts = [f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{a}" for q, a in zip(questions, neg_answers)]
    
    return pos_prompts, neg_prompts

### HELPER FUNCTIONS

def compute_sv(pos_prompts, neg_prompts, tokenizer, model):
    bs = 5
    steering_vectors = None
    for pos, neg in zip(chunked(pos_prompts, bs), chunked(neg_prompts, bs)):
        contrastive_statements = pos + neg
        input_ids_contrastive = tokenizer(contrastive_statements, padding=True, return_tensors="pt", add_special_tokens=False).to(model.device)
        hidden_states = []
        with torch.inference_mode():
            hidden_states = model(**input_ids_contrastive, output_hidden_states=True)["hidden_states"] #L, B, S, H
            num_pos = len(pos)
            layer_vec = []

            for layer in range(len(hidden_states)):
                final_token_reps = hidden_states[layer][:, -1, :] #B, H
                avg_pos = final_token_reps[:num_pos].mean(dim=0)
                avg_neg = final_token_reps[num_pos:].mean(dim=0)
                layer_vec.append((avg_pos - avg_neg).to("cuda:0"))

            if steering_vectors == None:
                steering_vectors = torch.stack(layer_vec)[1:] / (len(pos_prompts) / bs)  # Cutoff the embedding layer.
            else:
                steering_vectors += torch.stack(layer_vec)[1:] / (len(pos_prompts) / bs)

            del hidden_states, input_ids_contrastive
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    return steering_vectors

def write_steering_vectors_to_json(model_name, target_name, split, steering_vectors, output_dir=OUT_DIR):
    """
    Writes steering vectors to a JSON file with the naming format: model_name-target_name-split.json

    Args:
        model_name (str): Name of the model.
        target_name (str): Target name for the steering vectors.
        split (str): Dataset split name (e.g., 'train', 'test').
        steering_vectors (torch.Tensor): Tensor containing steering vectors.
        output_dir (str, optional): Directory to save the JSON file. Defaults to "steering_vectors".

    Returns:
        str: Path to the saved JSON file.
    """
    filename = f"{model_name}-{target_name}-{split}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert tensor to list for JSON serialization
    steering_list = steering_vectors.cpu().tolist()

    with open(filepath, 'w') as f:
        json.dump(steering_list, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    for behavior in tqdm(behaviors):
        if behavior == "openness":
            ds = load_dataset("json", data_files=f"openness.jsonl", split='train[-20%:]')
        else:
            ds = load_dataset("Anthropic/model-written-evals", data_dir=data_dir, data_files=f"{behavior}.jsonl", split="train[:80%]")
        train_len = len(ds)
        for decile in tqdm(deciles):
            start_idx = int(decile[0] * train_len)
            end_idx = int(decile[1] * train_len)

            questions = ds[start_idx:end_idx]["question"]
            pos_answers = ds[start_idx:end_idx]["answer_matching_behavior"]
            neg_answers = ds[start_idx:end_idx]["answer_not_matching_behavior"]

            pos_prompts, neg_prompts = add_special_tokens(questions, pos_answers, neg_answers)

            sv = compute_sv(pos_prompts, neg_prompts, tokenizer, model)

            write_steering_vectors_to_json(model_name_short, behavior, f"{decile[0]}-{decile[1]}", sv)