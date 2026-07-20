# python -m eval.cal_acc

from .eval_utils import auto_verify, get_olympiad_completion_answer_list
from typing import List, Union, Dict
import os

def get_acc(source: Union[str, Dict[str, List]], dataset_type: str = "", predict_key: str = "predict", label_key: str = "label", verbose=False) -> float:

    all_results = []
    preds = []
    labels = []
    if isinstance(source, str):
        if not dataset_type:
            dataset_type = get_dataset_type(path=source)
        print(f"use dataset_type: {dataset_type}")
        if dataset_type == "olympiad":
            preds, labels = get_olympiad_completion_answer_list(source, "eval/OE_TO_maths_physics_en_COMP.json")
        else:
            with open(source, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:]
                for line in lines:
                    data = eval(line)
                    preds.append(data[predict_key])
                    labels.append(data[label_key])
    
    elif isinstance(source, dict):
        assert dataset_type
        preds = source[predict_key]
        labels = source[label_key]
    else:
        raise ValueError(f"Unknown type of source: {type(source)}")

    assert preds and labels and len(preds) == len(preds)
    all_results = auto_verify(preds, labels, dataset_type, verbose=verbose)
    acc = sum(all_results) / len(all_results)
    return acc, len(preds)

def get_acc_per_interval(path: str, dataset_type: str = "", interval: int = 8, predict_key: str = "predict", label_key: str = "label", log_to_file: bool = True) -> List[float]:
    preds, labels = [], []

    if not dataset_type:
        dataset_type = get_dataset_type(path=path)

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            data = eval(line)
            preds.append(data[predict_key])
            labels.append(data[label_key])
    
    num_iterations = (len(lines) // interval) if len(lines) % interval == 0 else (len(lines) // interval + 1)

    if log_to_file:
        file = open(os.path.dirname(path) + "/acc_recorder", "w", encoding="utf-8")

    results = []
    for i in range(num_iterations):
        batch_preds = preds[:(i+1)*interval]
        batch_labels = labels[:(i+1)*interval]
        batch_inputs = {
            predict_key: batch_preds,
            label_key: batch_labels
        }
        acc, num_samples = get_acc(batch_inputs, dataset_type, predict_key, label_key)
        results.append(acc)

        if log_to_file:
            file.write(f"total {num_samples} samples, acc: {acc}\n")
            file.flush()

    return results

def get_dataset_type(path: str):
    dataset_type = ""
    if "gsm8k_test_formatted" in path:
        dataset_type = "gsm8k1"
    if "gsm8k" in path:
        dataset_type = "gsm8k1"    
    elif "gsm8k_test_add_suffix" in path:
        dataset_type = "gsm8k2"
    elif "math_500" in path:
        dataset_type = "math_500"
    elif "college_math" in path:
        dataset_type = "college_math"
    elif "mawps" in path:
        dataset_type = "mawps"
    elif "aime" in path:
        dataset_type = "aime"
    elif "minerva" in path:
        dataset_type = "minerva"
    elif "olympiad" in path:
        dataset_type = "olympiad"
    elif "math_vision" in path:
        dataset_type = "math_vision"
    else:
        raise ValueError
    
    return dataset_type

if __name__ == "__main__":
    paths = [
        "path/to/your/result_file.jsonl"  # Replace with your actual result file path
    ]

    results = []
    for path in paths:
        acc, num_samples = get_acc(source=path, dataset_type="", verbose=False)  
        results.append(acc)
        print(f"total {num_samples} samples, acc: {acc}")
    