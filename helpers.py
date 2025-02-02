import torch
import os
import json

def create_steering_vectors(pos_prompts, neg_prompts, tokenizer, model):
    contrastive_statements = pos_prompts + neg_prompts
    input_ids_contrastive = tokenizer(contrastive_statements, padding=True, return_tensors="pt").to("cuda")
    
    steering_vectors = []
    
    with torch.no_grad():
        hidden_states = model(**input_ids_contrastive, output_hidden_states=True)["hidden_states"] #L, B, S, H
        num_pos = len(pos_prompts)
        num_neg = len(neg_prompts)
        
        for layer in range(len(hidden_states)):
            final_token_reps = hidden_states[layer][:, -1, :].float() #B, H
            avg_pos = final_token_reps[:num_pos].mean(dim=0)
            avg_neg = final_token_reps[num_pos:num_pos+num_neg].mean(dim=0) #B, H
            steering_vectors.append(avg_pos - avg_neg)
    
        steering_vectors = torch.stack(steering_vectors)[1:]  # Cutoff the embedding layer.
    
    return steering_vectors


def get_mwe_subset(ds, start_id, end_id, file_name):
    """
    file_name: str - The name of the file in the dataset
    """
    text_slice = ds[file_name][start_id:end_id]["question"]  # Slices the 'train' split
    return text_slice


def write_steering_vectors_to_json(model_name, target_name, split, steering_vectors, output_dir="steering_vectors"):
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
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, filename)

    # Convert tensor to list for JSON serialization
    steering_list = steering_vectors.cpu().tolist()

    with open(filepath, 'w') as f:
        json.dump(steering_list, f, ensure_ascii=False, indent=4)