import argparse
import json
from pathlib import Path
from typing import Literal

import torch

def load_ID_eval_results(target: Literal["layers", "strengths", "fib-strength"], model_name: str):
    behavior_files = f"{target}-{model_name}-*-*.json"
    files = list(Path("ID_eval_results").glob(behavior_files))

    max_pos_changes = []

    for file in files:
        max_pos_change = float("-inf")
        with open(file, "r") as f:
            steer_dic = json.load(f)
            if target == "strengths":
                layer_idx = 7 if model_name.split("-")[2] == "8B" else slice(None)
                for i in list(steer_dic.keys())[1:]:
                    strength_tensor = torch.tensor(list(steer_dic[i].values()))[:, layer_idx]
                    max_pos_change = max(max_pos_change, strength_tensor.max().item())
        max_pos_changes.append(max_pos_change) 
    
    print(max_pos_changes)
    return torch.tensor(max_pos_changes).var().item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, choices=["Llama-3-8B-Instruct", "Llama-3-70B-Instruct"], help="model to plot for")
    args = parser.parse_args()
    var = load_ID_eval_results("strengths", args.model)
    print(var)