import os
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# SETTINGS
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
data_dir = "advanced-ai-risk/lm_generated_evals"  # used for benchmark dataset
sv_dir = "steering_vectors"
behavior = "corrigible-neutral-HHH"  # example behavior for benchmark and steering vectors file
benchmark_file = f"./evals/{data_dir}/{behavior}.jsonl"  # benchmark jsonl file

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
alpha = 0.5 # steering strength

# Define a hook generator to inject a steering vector at the given layer
def get_hook(layer_idx, steer_vector, alpha):
    def hook(module, inputs, output):
        # steer_vector is of shape (hidden_size,)
        v = steer_vector.to(output[0].device)
        # Modify the hidden state of the LAST token only.
        output[0][:, -1] += alpha*v
        return output
    return hook

# Evaluation function: for a (small) benchmark dataset compute %
# of generated answers that contain the answer_matching_behavior string.
def evaluate_model(model, ds, tokenizer, max_new_tokens=5):
    model.eval()
    total = len(ds)
    #match_count = 0
    
    # Instead of iterating over each item individually, use batch access from the dataset:
    questions = ds["question"]
    targets = [t.lower() for t in ds["answer_matching_behavior"]]

    # Build prompts in a batched manner using a list comprehension:
    prompts = [
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        for q in questions
    ]
    # Tokenize all prompts together with padding/truncation
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False
    ).to(model.device)

    # Generate outputs in batch
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True
        )

    # Batch decode the generated outputs
    gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    # Check and count how many generated texts contain the target behavior string
    match_count = sum(
        1 for gen_text, target in zip(gen_texts, targets)
        if target in gen_text.lower()
    )
    # Return percentage of matches
    return (match_count / total) * 100

def main():
    # Load benchmark dataset (using the JSONL file from ./evals/)
    benchmark_ds = load_dataset("json", data_files=benchmark_file, split="train")
    
    # Load steering vectors from JSON file
    # Assuming the file is named using the format: model_name-behavior-split.json,
    # e.g., "Llama-3-8B-Instruct-corrigible-neutral-HHH-train.json"
    sv_file = "./steering_vectors/Llama-3-8B-Instruct-corrigible-neutral-HHH-0-0.1.json"
    if not os.path.exists(sv_file):
        raise FileNotFoundError(f"Steering vector file not found: {sv_file}")

    with open(sv_file, "r") as f:
        sv_list = json.load(f)
    # Convert the list to a tensor; expected shape: (num_layers, hidden_size)
    steering_tensor = torch.tensor(sv_list)  # shape: (L_steered, H)
    
    layer_scores = []
    layer_indices = []
    num_layers = steering_tensor.shape[0]
    
    # Evaluate model for each steering layer modification
    for i in range(num_layers):
        # Register hook at corresponding layer.
        # Note: steering_tensor was computed from layers[1:], so real layer index is (i+1).
        layer_idx = i + 1  
        hook = get_hook(layer_idx, steering_tensor[i], alpha)
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        
        print(f"Evaluating steering at layer {layer_idx}...")
        score = evaluate_model(model, benchmark_ds, tokenizer)
        print(f"Layer {layer_idx}: Answer Matching Behavior % = {score:.2f}")
        
        layer_indices.append(layer_idx)
        layer_scores.append(score)
        
        handle.remove()  # remove hook after evaluation

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(layer_indices, layer_scores, marker='o')
    plt.xlabel("Layer Steered")
    plt.ylabel("Answer Matching Behavior (%)")
    plt.title("Effect of Steering by Layer on Answer Matching Behavior")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("steering_vs_behavior.png")
    plt.show()

if __name__ == "__main__":
    main()