import argparse
import os
import json
from pathlib import Path
from typing import Literal
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch

def merge_strength_results(model_name: str):
    behavior_files = f"strengths-{model_name}-*.json"
    files = sorted(Path("ID_eval_results").glob(behavior_files))
    result_dic = {}
    for file in files:
        with open(file, "r") as f:
            file_dic = json.load(f)
            for ok, d in file_dic.items():
                if ok not in result_dic:
                    result_dic[ok] = d
                else:
                    for ik, v in d.items():
                        if ik in result_dic[ok]:
                            result_dic[ok][ik].extend(v)
                        else:
                            result_dic[ok][ik] = v
    return result_dic

def merge_mmlu_results(model_name: str):
    behavior_files = f"{model_name}-*.json"
    files = sorted(Path("MMLU_eval_results").glob(behavior_files))
    result_dic = {}
    for file in files:
        with open(file, "r") as f:
            file_dic = json.load(f)
            for ok, d in file_dic.items():
                if ok not in result_dic:
                    result_dic[ok] = d
                else:
                    if isinstance(d, dict):
                        for ik, v in d.items():
                            if ik in result_dic[ok]:
                                result_dic[ok][ik].extend(v)
                            else:
                                result_dic[ok][ik] = v
                    else:
                        pass
    return result_dic



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, choices=["Llama-3-8B-Instruct", "Llama-3.1-70B-Instruct"], help="model to plot for")
    parser.add_argument("-r", "--results", type=str, required=True, choices=["strengths", "mmlu"], help="results to aggregate")
    parser.add_argument("-d", "--dir", type=str, required=True, help="directory to output results to")
    args = parser.parse_args()

    if args.results == "strengths":
        result = merge_strength_results(args.model)
    else:
        result = merge_mmlu_results(args.model)

    os.makedirs(args.dir, exist_ok=True)
    out_filepath = os.path.join(
        args.dir,
        f"{args.model}.json" 
    )
    with open(out_filepath, "w") as f:
        json.dump(result, f, indent=4)