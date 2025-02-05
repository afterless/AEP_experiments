from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
print(model.generation_config)