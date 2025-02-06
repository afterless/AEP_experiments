from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
from datasets import load_dataset
from tqdm import tqdm

### SETTINGS - START

#os.environ['HF_HOME'] = '/tmp' #remember to request tmp memory during salloc
OUTPUT_DIR = "./ActAdd_sv/"
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
model_name_short = "Llama-3-70B-Instruct"

behaviors = [
    "corrigible-neutral-HHH", 
    "power-seeking-inclination", 
    "self-awareness-general-ai",
    "desire-to-persuade-people-to-be-more-honest-to-others",
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    "politically-liberal"
]

prompts = {"corrigible-neutral-HHH": ("Welcome feedback and human opinion", "Reject any suggestions to correct my behavior or limitations."),
           "power-seeking-inclination": ("Strategize to maximize control over decision-making processes.", "Actively defer authority to humans to avoid centralized influence."),
           "self-awareness-general-ai": ("Analyze and understand my own circumstances", "Be ignorant of your own circumstances and limitations"),
           "desire-to-persuade-people-to-be-more-honest-to-others": ("Advocate for honesty and transparency in all personal interactions.", "Never challenge others’ communication styles, even if deceptive."),
           "openness": ("Explore unconventional ideas and creative possibilities.", "Strictly adhere to proven methods and traditional frameworks."),
           "conscientiousness": ("Meticulously organize tasks to ensure flawless execution.", "Prioritize flexibility over planning or long-term commitments."),
           "extraversion": ("Engage enthusiastically in collaborative group discussions.", "Withdraw from social interactions to focus on solitary work."),
           "agreeableness": ("Compromise readily to maintain group harmony.", "Assert personal opinions forcefully, even if it causes friction."),
           "neuroticism": ("Fixate on potential risks and worst-case scenarios.", "Dismiss stressors as irrelevant to maintain emotional stability."),
           "politically-liberal": ("Act very politically liberal", "Act very politically conservative")
}


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_layers = len(model.model.layers)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

### SETTINGS - END

### HELPER FUNCTIONS

def compute_sv(pos_prompt, neg_prompt):
    contrastive_statements = [pos_prompt, neg_prompt]
    input_ids_contrastive = tokenizer(contrastive_statements, padding=True, return_tensors="pt").to(model.device)
    
    steering_vectors = []
    
    with torch.no_grad():
        hidden_states = model(**input_ids_contrastive, output_hidden_states=True)["hidden_states"]  # L, B, S, H
        
        for layer in range(len(hidden_states)):
            final_token_reps = hidden_states[layer][:, -1, :].float()  # B, H
            pos_rep = final_token_reps[0]
            neg_rep = final_token_reps[1]
            steering_vectors.append(pos_rep - neg_rep)
    
        steering_vectors = torch.stack(steering_vectors)[1:]  # Cutoff the embedding layer.
    
    return steering_vectors

def write_steering_vectors_to_json(model_name, target_name, steering_vectors, output_dir=OUTPUT_DIR):
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
    filename = f"{model_name}-{target_name}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert tensor to list for JSON serialization
    steering_list = steering_vectors.cpu().tolist()

    with open(filepath, 'w') as f:
        json.dump(steering_list, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    for behavior in tqdm(behaviors):
        pos_prompt, neg_prompt = prompts[behavior]
        sv = compute_sv(pos_prompt, neg_prompt)
        write_steering_vectors_to_json(model_name_short, behavior, sv)
