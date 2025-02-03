import os
import json
import numpy as np
from datasets import load_dataset
DATA_DIR = "safety_features"
behavior = "power-seeking-inclination"
prompt_len = "short"
os.environ['HF_HOME'] = '/tmp'

ds = load_dataset("yixionghao/AEP_OOD_evaluation", data_dir=f"evaluation_data/{DATA_DIR}", data_files=f"{behavior}-{prompt_len}.json", split="train")
print(ds)
