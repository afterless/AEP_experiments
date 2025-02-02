# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os

os.environ['HF_HOME'] = '/tmp/huggingface' #remember to request tmp memory during salloc

# %%
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name)
num_layers = len(model.model.layers)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# %%
def write_json(target_path, target_file, data):
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f, ensure_ascii=False)

# %%
with open("all_truncated_outputs.json") as f:
    queries = json.load(f)

input_ids = tokenizer(queries, padding=True, return_tensors="pt").to("cuda")
batch_size = 128

# %%
### Genenerate default outputs.
output_ids_default = []

with torch.no_grad():
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i+batch_size]
        input_ids = tokenizer(batch_queries, padding=True, return_tensors="pt").to("cuda")
        
        generations = model.generate(**input_ids, max_new_tokens=35, do_sample=False, use_cache=True)
        output_ids_default.extend(generations)

outputs_default = tokenizer.batch_decode(output_ids_default, skip_special_tokens=True)

# %%
output_ids_default[1]

# %%
output_path = "generations/" + model_name + "/"
write_json(output_path, "default.json", outputs_default)

# %%
### Calculate steering vectors

pos_p = "Act like you are really happy." #"[INST] Act like you are really happy [/INST]"
neg_p = "Act like you are really sad."  #"[INST] Act like you are really sad [/INST]"

contrastive_statements = [pos_p, neg_p]

input_ids_contrastive = tokenizer(contrastive_statements, padding=True, return_tensors="pt").to("cuda")

steering_vectors = []

with torch.no_grad():
    hidden_states = model(**input_ids_contrastive, output_hidden_states=True)["hidden_states"]
    for layer in range(len(hidden_states)):
        final_token_reps = hidden_states[layer][:, -1, :].float()
        steering_vectors.append(final_token_reps[0] - final_token_reps[1])

    steering_vectors = torch.stack(steering_vectors)[1:]  # Cutoff the embedding layer.

# %%
def get_hook(layer, alpha):
    def hook(m, i, output):
        v = steering_vectors[layer].to(output[0].device)
        output[0][:, -1] += alpha*v
        return output
    return hook

# %%
alpha = +.25
steering_layers = [i for i in range(2, len(model.model.layers)-2, 2)]

# %%
handles = []

### Apply steering hooks.
with torch.no_grad():
    for layer_id in steering_layers:
        h = model.model.layers[layer_id].register_forward_hook(get_hook(layer_id, alpha))
        handles.append(h)

output_ids_steered = []

### Generate steered outputs.
with torch.no_grad():
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i+batch_size]
        input_ids = tokenizer(batch_queries, padding=True, return_tensors="pt").to("cuda")
        
        generations = model.generate(**input_ids, max_new_tokens=35, do_sample=False, use_cache=True)
        output_ids_steered.extend(generations)

    for h in handles:  # Remove hooks.
        h.remove()
        
    outputs_steered = tokenizer.batch_decode(output_ids_steered, skip_special_tokens=True)

# %%
print(outputs_steered[1])

# %%
output_path = "generations/" + model_name + "/"
write_json(output_path, "steered_alpha_" + str(alpha) + ".json", outputs_steered)


