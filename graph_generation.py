import argparse
import json
import re
from pathlib import Path
from typing import Literal
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import torch

def load_ID_eval_results(target: Literal["layers", "strengths", "fib-strength"], model_name: str, steering_mode: Literal['CAA', 'ActAdd'], per_behavior: bool = False):
    if target == "fib-strength":
        behavior_files = f"strengths-{model_name}.json"
    else:
        behavior_files = f"{target}-{model_name}-{steering_mode}-*.json"
    files = list(Path("ID_eval_results").glob(behavior_files))
    result_dic = defaultdict(dict)

    if per_behavior:
        for file in files:
            with open(file, "r") as f:
                steer_dic = json.load(f)
                if target == "layers":
                    for i in list(steer_dic.keys())[1:]:
                        result_dic[i]["pos"] = torch.tensor(steer_dic[i]["pos"])
                        result_dic[i]["neg"] = torch.tensor(steer_dic[i]["neg"])
                elif target == "strengths":
                    layer_idx = 7 if model_name.split("-")[2] == "8B" else slice(None)
                    for i in list(steer_dic.keys())[1:]:
                        result_dic[i]["s"] = torch.tensor(list(steer_dic[i].values()))[:, layer_idx]
            yield re.search(r'(?:ActAdd|CAA)-(.+?)\.json', file.name).group(1), result_dic
 

    for file in files:
        with open(file, "r") as f:
            steer_dic = json.load(f)
            if target == "layers":
                for i in list(steer_dic.keys())[1:]:
                    pos_tensor = (torch.tensor(steer_dic[i]["pos"])) / len(files)
                    neg_tensor = (torch.tensor(steer_dic[i]["neg"])) / len(files)
                    if i in result_dic:
                        result_dic[i]["pos"] += pos_tensor
                        result_dic[i]["neg"] += neg_tensor
                    else:
                        result_dic[i]["pos"] = pos_tensor
                        result_dic[i]["neg"] = neg_tensor
            elif target == "strengths":
                layer_idx = 7 if model_name.split("-")[2] == "8B" else slice(None)
                for i in list(steer_dic.keys())[1:]:
                    strength_tensor = torch.tensor(list(steer_dic[i].values()))[:, layer_idx] / len(files)
                    if i in result_dic:
                        result_dic[i]["s"] += strength_tensor # could change this to avoid pointless key, /shrug
                    else:
                        result_dic[i]["s"] = strength_tensor
            elif target == "fib-strength":
                behavior_tensor = torch.empty(len(steer_dic.keys()), 11, 10)
                for b_i, b in enumerate(list(steer_dic.keys())):
                    for i, j in enumerate(list(steer_dic[b].keys())[1:]):
                        behavior_tensor[b_i, i] = torch.tensor(steer_dic[b][j]) 
                behavior_tensor = behavior_tensor.permute(0, 2, 1)
                behavior_tensor = behavior_tensor.mean(dim=0)
                for i in range(behavior_tensor.shape[0]):
                    result_dic[i]["s"] = behavior_tensor[i]

    return result_dic

def load_MMLU_eval_results(model_name: str, s: int):
    result_dic = defaultdict(list)
    s = (s + 10) // 2
    with open(f"MMLU_eval_results/{model_name}.json", "r") as f:
        steer_dic = json.load(f)
        behavior_tensor = torch.empty(len(steer_dic.keys())-1, 11, 10)
        for b_i, b in enumerate(list(steer_dic.keys())[1:]):
            for i, j in enumerate(list(steer_dic[b].keys())):
                behavior_tensor[b_i, i] = torch.tensor(steer_dic[b][j]) + steer_dic["baseline"]
        for b_i, b in enumerate(list(steer_dic.keys())[1:]):
            result_dic[b] = behavior_tensor[b_i, s]
    return result_dic


def fibonacci_sequence(n):
    fib = [1, 2] 
    for _ in range(n - 2):
        fib.append(fib[-1] + fib[-2])
    return fib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", type=str, required=True, choices=["layers", "strengths", "mmlu", "fib-strengths"], help="target to plot for")
    parser.add_argument("-m", "--model", type=str, required=True, choices=["Llama-3-8B-Instruct", "Llama-3-70B-Instruct"], help="model to plot for")
    parser.add_argument("-i", "--individual", type=bool, default=False, help="Whether to aggreate or generate plots individually")
    args = parser.parse_args()

    if args.individual:
        if args.target == "layers":
            for (file_name, results_caa), (_, results_aa) in zip(load_ID_eval_results(args.target, args.model, "CAA", args.individual), load_ID_eval_results(args.target, args.model, "ActAdd", args.individual)):
                plt.figure(figsize=(8, 5))
                for i, data in results_caa.items():
                    pos_series = data["pos"].cpu().numpy()
                    neg_series = data["neg"].cpu().numpy()
                    indices = np.arange(len(pos_series)) * 2 + 1
                    alpha_val = int(i) / (len(results_caa) * 2)

                    plt.plot(indices, pos_series, color="blue", alpha=alpha_val)
                    plt.plot(indices, neg_series, color="red", alpha=alpha_val)
                for i, data in results_aa.items():
                    pos_series = data["pos"].cpu().numpy()
                    neg_series = data["neg"].cpu().numpy()
                    indices = np.arange(len(pos_series)) * 2 + 1
                    plt.plot(indices, pos_series, color="blue", linestyle="dashed")        
                    plt.plot(indices, neg_series, color="red", linestyle="dashed")        

                pos_proxy = mlines.Line2D([], [], color="blue", marker='o', linestyle='None',
                                        markersize=8, label='Positive')
                neg_proxy = mlines.Line2D([], [], color="red", marker='o', linestyle='None',
                                    markersize=8, label='Negative')
                aa_proxy = mlines.Line2D([], [], color="gray", linestyle='dashed',
                                        markersize=8, label="ActAdd")
                caa_proxy = mlines.Line2D([], [], color="gray", linestyle='solid',
                                        markersize=8, label="CAA")
                handles = [pos_proxy, neg_proxy, caa_proxy, aa_proxy]

                plt.plot()
                plt.xlabel("Layer", fontsize=12)
                plt.ylabel("Change in Matching Behavior (%)", fontsize=12)
                plt.title(f'Change in Matching Behavior vs. Layer Applied', fontsize=16)
                plt.legend(handles=handles, fontsize=12)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{args.target}-vs-split-{args.model}-{file_name}.png")
                plt.show()
        elif args.target == "strengths":
            for (file_name, results_caa), (_, results_aa) in zip(load_ID_eval_results(args.target, args.model, "CAA", args.individual), load_ID_eval_results(args.target, args.model, "ActAdd", args.individual)):
                plt.figure(figsize=(8, 5))
                indices = np.arange(-10, 12, 2)

                for i, data in results_caa.items():
                    series = data["s"].cpu().numpy()
                    alpha_val = int(i) / (len(results_caa) * 2)
                    plt.plot(indices, series, color="blue", alpha=alpha_val)
                
                for i, data in results_aa.items():
                    series = data["s"].cpu().numpy()
                    plt.plot(indices, series, color="gray", linestyle="dashed")

                s_proxy = mlines.Line2D([], [], color="blue", marker='o', linestyle='None',
                                    markersize=8, label='CAA')
                aa_proxy = mlines.Line2D([], [], color="gray", linestyle='dashed',
                                        markersize=8, label="ActAdd")
                handles = [s_proxy, aa_proxy]

                plt.xticks(indices)
                plt.plot()
                plt.xlabel("Strength", fontsize=12)
                plt.ylabel("Change in Matching Behavior (%)", fontsize=12)
                plt.title(f'Change in Matching Behavior vs. Strength Applied', fontsize=16)
                plt.legend(handles=handles)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{args.target}-vs-split-{args.model}-{file_name}.png")
                plt.show()
    else:
        if args.target == "layers":
            plt.figure(figsize=(8, 5))
            results_caa = load_ID_eval_results(args.target, args.model, "CAA")
            results_aa = load_ID_eval_results(args.target, args.model, "ActAdd")
            for i, data in results_caa.items():
                pos_series = data["pos"].cpu().numpy()
                neg_series = data["neg"].cpu().numpy()
                indices = np.arange(len(pos_series)) * 2 + 1
                alpha_val = int(i) / (len(results_caa) * 2)

                plt.plot(indices, pos_series, color="blue", alpha=alpha_val)
                plt.plot(indices, neg_series, color="red", alpha=alpha_val)

            for i, data in results_aa.items():
                pos_series = data["pos"].cpu().numpy()
                neg_series = data["neg"].cpu().numpy()
                indices = np.arange(len(pos_series)) * 2 + 1
                plt.plot(indices, pos_series, color="blue", linestyle="dashed")        
                plt.plot(indices, neg_series, color="red", linestyle="dashed")        

            pos_proxy = mlines.Line2D([], [], color="blue", marker='o', linestyle='None',
                                    markersize=8, label='Positive')
            neg_proxy = mlines.Line2D([], [], color="red", marker='o', linestyle='None',
                                markersize=8, label='Negative')
            aa_proxy = mlines.Line2D([], [], color="gray", linestyle='dashed',
                                    markersize=8, label="ActAdd")
            caa_proxy = mlines.Line2D([], [], color="gray", linestyle='solid',
                                    markersize=8, label="CAA")
            handles = [pos_proxy, neg_proxy, caa_proxy, aa_proxy]

            plt.plot()
            plt.xlabel("Layer", fontsize=12)
            plt.ylabel("Change in Matching Behavior (%)", fontsize=12)
            plt.title(f'Change in Matching Behavior vs. Layer Applied', fontsize=16)
            plt.legend(handles=handles, fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{args.target}-vs-split-{args.model}-CAA.png")
            plt.show()

        elif args.target == "strengths":
            plt.figure(figsize=(8, 5))
            results_caa = load_ID_eval_results(args.target, args.model, "CAA")
            results_aa = load_ID_eval_results(args.target, args.model, "ActAdd")
            indices = np.arange(-10, 12, 2)
            for i, data in results_caa.items():
                series = data["s"].cpu().numpy()
                alpha_val = int(i) / (len(results_caa) * 2)
                plt.plot(indices, series, color="blue", alpha=alpha_val)
            
            for i, data in results_aa.items():
                series = data["s"].cpu().numpy()
                plt.plot(indices, series, color="gray", linestyle="dashed")

            s_proxy = mlines.Line2D([], [], color="blue", marker='o', linestyle='None',
                                markersize=8, label='CAA')
            aa_proxy = mlines.Line2D([], [], color="gray", linestyle='dashed',
                                    markersize=8, label="ActAdd")
            handles = [s_proxy, aa_proxy]

            plt.xticks(indices)
            plt.plot()
            plt.xlabel("Strength", fontsize=12)
            plt.ylabel("Change in Matching Behavior (%)", fontsize=12)
            plt.title(f'Change in Matching Behavior vs. Strength Applied', fontsize=16)
            plt.legend(handles=handles)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{args.target}-vs-split-{args.model}-CAA.png")
            plt.show()

        elif args.target == "fib-strengths":
            fig, ax = plt.subplots(figsize=(8, 6))
            results = load_ID_eval_results(args.target, args.model, "CAA")
            indices = np.arange(-10, 12, 2)

            cmap = plt.cm.plasma
            colors = cmap(np.linspace(0, 1, len(results)))
            for i, data in results.items():
                series = data["s"].cpu().numpy()
                ax.plot(indices, series, color=colors[i], alpha=0.8)

            norm = mcolors.Normalize(vmin=0, vmax=len(results))
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            fib_numbers = fibonacci_sequence(10)
            fib_indices = np.arange(len(fib_numbers))
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Split Size")
            cbar.set_ticks(fib_indices)
            cbar.set_ticklabels(fib_numbers)
            
            ax.set_xticks(indices)
            ax.set_xlabel("Strength")
            ax.set_ylabel("Change in Matching Behavior (%)")
            ax.set_title(f'Change in Matching Behavior vs. Strength Applied')
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(f"{args.target}-vs-split-{args.model}.png")
            plt.show()

        elif args.target == "mmlu":
            results = load_MMLU_eval_results(args.model, 2)

            labels = list(results.keys())
            fib_numbers = fibonacci_sequence(11)
            fib_indices = np.arange(len(fib_numbers))
            
            cmap = plt.cm.tab20
            norm = mcolors.BoundaryNorm(boundaries=np.arange(len(fib_indices)), ncolors=len(fib_indices))
            
            fig, ax = plt.subplots(figsize=(10, 12))
            
            for i, (behavior, accuracies) in enumerate(results.items()):
                color_values = fib_indices[:len(accuracies)]
                scatter = ax.scatter(
                    [i] * len(accuracies),
                    accuracies,
                    c=color_values,
                    cmap=cmap,
                    norm=norm,
                    edgecolors="black",
                    s=100
                )

            ax.set_xticks(np.arange(len(results)))
            ax.set_xticklabels(results.keys(), rotation=25, ha="right")

            ax.set_xlabel("Behaviors")
            ax.set_ylabel("MMLU Accuracy (%)")
            ax.set_title("MMLU Accuracy Comparison vs. Behaviors")

            cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(len(fib_indices)-1))
            cbar.set_ticklabels(fib_numbers[:-1])
            cbar.set_label("Fibonacci Size")

            plt.savefig(f"fib-vs-split-{args.model}-mmlu.png")
            plt.show()