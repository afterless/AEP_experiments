import os
import json
import torch

# Adjust the file path to one of your steering_vectors JSON files.
# For example, if your file is saved as "Llama-3-8B-Instruct-corrigible-neutral-HHH-0.0-0.1.json" in the steering_vectors directory:
filepath = os.path.join("steering_vectors", "Llama-3-8B-Instruct-corrigible-neutral-HHH-0-0.1.json")

with open(filepath, "r") as f:
    steering_data = json.load(f)

# Convert the loaded list into a tensor
steering_tensor = torch.tensor(steering_data)

# Print the dimensions
print("Steering vectors tensor shape:", steering_tensor.shape)
