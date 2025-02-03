import os
import json
import argparse
import numpy as np

deciles = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
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
input_dir = "./steering_vectors"
output_dir = "./complete_sv"

def main():
    # Get list of JSON files in the input directory.
    for behavior in behaviors:
        total = None
        count = 0
        for decile in deciles:
            file_name = os.path.join(input_dir, f"Llama-3-8B-Instruct-{behavior}-{decile[0]}-{decile[1]}.json")

            if os.path.exists(file_name):
                with open(file_name, "r") as f:
                    data = json.load(f)
                # Convert the JSON list to a numpy array (tensor)
                tensor = np.array(data)
                if behavior == "corrigible-neutral-HHH" and decile == (0.0, 0.1):
                    print(tensor.shape)
                if total is None:
                    total = np.zeros_like(tensor)
                # Element-wise addition of the tensor
                total += tensor
                count += 1

        if count > 0:
            avg = (total / count).tolist()
            if behavior == "corrigible-neutral-HHH":
                print(len(avg))
                print(len(avg[0]))
            # Write the averaged tensor to a JSON file
            with open(f"./{output_dir}/Llama-3-8B-Instruct-{behavior}.json", "w") as out_file:
                json.dump(avg, out_file, indent=4)
            print(f"Averaged steering vectors saved to merged_vectors.json")
        else:
            print("No valid steering vector files were found.")

if __name__ == "__main__":
    main()