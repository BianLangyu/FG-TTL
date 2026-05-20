# python -m eval.cal_acc

from .eval_utils import auto_verify, get_olympiad_completion_answer_list
from typing import List, Union, Dict
import os

def get_acc(source: Union[str, Dict[str, List]], dataset_type: str = "", predict_key: str = "predict", label_key: str = "label", verbose=False) -> float:
    """传入模型生成结果的文件路径或数据字典，返回准确率"""
    # assert dataset_type in ["gsm8k1", "gsm8k2", "math_500", "mawps", "college_math", "aime"], f"Unknown dataset_type: {dataset_type}"
    
    all_results = []
    preds = []
    labels = []
    if isinstance(source, str):
        if not dataset_type:
            dataset_type = get_dataset_type(path=source)
        print(f"use dataset_type: {dataset_type}")
        # 读文件，收集数据
        if dataset_type == "olympiad":
            preds, labels = get_olympiad_completion_answer_list(source, "FG-TTL/data/LLMTTA/OlympiadBench/OE_TO_maths_physics_en_COMP.json")
        else:
            with open(source, 'r', encoding='utf-8') as f:
                lines = f.readlines()
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
    # 计算准确率
    all_results = auto_verify(preds, labels, dataset_type, verbose=verbose)
    acc = sum(all_results) / len(all_results)
    return acc, len(preds)

def get_acc_per_interval(path: str, dataset_type: str = "", interval: int = 8, predict_key: str = "predict", label_key: str = "label", log_to_file: bool = True) -> List[float]:
    """每隔 interval 个样本就算一次准确率"""
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
            file.write(f"总共 {num_samples} 个样本，准确率为：{acc}\n")
            file.flush()

    return results

def get_dataset_type(path: str):
    """从路径里解析出dataset_type"""
    dataset_type = ""
    if "gsm8k_test_formatted" in path:
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
    else:
        raise ValueError
    
    return dataset_type

if __name__ == "__main__":
    paths = [
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.2-3B-Instruct/online_tent/gsm8k_test_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/gsm8k_test_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.2-3B-Instruct/online_tent/aime25_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/math_500/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/mawps_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/mawps_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/aime25_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory/saves/emnlp2025/llama3.1-8b/online_tent/gsm8k_test_formatted-one_lora/first_few_token/naive_entropy-diff_lora_init_0526/lr_5e-6-new_tokens_8-seed_42/predict-temperature_0.0-max_new_tokens_1024/generated_predictions.jsonl",
        
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-2-13b-chat-hf/online_tent/gsm8k_test_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_3072/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-2-13b-chat-hf/online_tent/math_500/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_3072/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-2-13b-chat-hf/online_tent/mawps_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_3072/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-2-13b-chat-hf/online_tent/college_math_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_3072/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-2-13b-chat-hf/online_tent/aime24_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_3072/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-2-13b-chat-hf/online_tent/aime25_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_3072/generated_predictions.jsonl"
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-7B-Instruct/online_tent/gsm8k_test_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-7B-Instruct/online_tent/math_500/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-7B-Instruct/online_tent/mawps_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-7B-Instruct/online_tent/college_math_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-7B-Instruct/online_tent/aime24_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-7B-Instruct/online_tent/aime25_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/online_tent/gsm8k_test_add_suffix/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/online_tent/math_500/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/online_tent/mawps_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/online_tent/college_math_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/online_tent/aime24_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/online_tent/aime25_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_32-K_max_24-K_min_2-initial_delta_0.01-patient_m_2-threshold_low_0.02-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_32-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-clip_high_0.01-clip_low_0.002-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_32-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-clip_high_0.02-clip_low_0.001-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_32-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-clip_high_0.02-clip_low_0.002-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_32-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_32-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-std_coef_0.01-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_32-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-std_coef_0.05-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_32-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-std_coef_0.005-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_32-K_max_24-K_min_2-initial_delta_0.02-patient_m_3-threshold_low_0.02-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_32-K_max_24-K_min_2-initial_delta_0.02-patient_m_4-threshold_low_0.02-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_32-K_max_24-K_min_2-initial_delta_0.02-patient_m_5-threshold_low_0.02-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_32-K_max_24-K_min_2-initial_delta_0.03-patient_m_2-threshold_low_0.02-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_2-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_4-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        
        
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_8-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_16-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/gsm8k_test_formatted/naive_entropy/lr_5e-6-new_tokens_32-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory/saves/emnlp2025/llama3.1-8b/online_tent/gsm8k_test_formatted-one_lora/first_few_token/SAFE_Entropy_loss/lr_5e-6-new_tokens_8-seed_42/predict-temperature_0.0-max_new_tokens_1024/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory/saves/emnlp2025/llama3.1-8b/online_tent/gsm8k_test_formatted-one_lora/first_few_token/SAFE_Entropy_loss/lr_5e-6-new_tokens_16-seed_42/predict-temperature_0.0-max_new_tokens_1024/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory/saves/emnlp2025/llama3.1-8b/online_tent/gsm8k_test_formatted-one_lora/first_few_token/SAFE_Entropy_loss/lr_5e-6-new_tokens_32-seed_42/predict-temperature_0.0-max_new_tokens_1024/generated_predictions.jsonl",
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/gsm8k_test_formatted/SAFE_Entropy_loss_v2/lr_5e-6-new_tokens_8-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/gsm8k_test_formatted/SAFE_Entropy_loss_v2/lr_5e-6-new_tokens_16-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/gsm8k_test_formatted/SAFE_Entropy_loss_v2/lr_5e-6-new_tokens_32-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/math_500/naive_entropy/lr_5e-6-new_tokens_8-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/math_500/naive_entropy/lr_5e-6-new_tokens_16-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/math_500/naive_entropy/lr_5e-6-new_tokens_32-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/math_500/SAFE_Entropy_loss/lr_5e-6-new_tokens_8-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/math_500/SAFE_Entropy_loss/lr_5e-6-new_tokens_16-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/math_500/SAFE_Entropy_loss/lr_5e-6-new_tokens_32-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/math_500/SAFE_Entropy_loss_v2/lr_5e-6-new_tokens_8-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/math_500/SAFE_Entropy_loss_v2/lr_5e-6-new_tokens_32-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/base_model_results-vllm/gsm8k_test_add_suffix/gsm8k_test_add_suffix-template_None-temperature_0.0-max_new_tokens_8192-1319samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/base_model_results-vllm/math_500/math_500-template_None-temperature_0.0-max_new_tokens_8192-500samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/base_model_results-vllm/mawps_formatted/mawps_formatted-template_None-temperature_0.0-max_new_tokens_8192-238samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/base_model_results-vllm/college_math_formatted/college_math_formatted-template_None-temperature_0.0-max_new_tokens_8192-1200samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/base_model_results-vllm/aime24_formatted/aime24_formatted-template_None-temperature_0.0-max_new_tokens_8192-30samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/base_model_results-vllm/aime25_formatted/aime25_formatted-template_None-temperature_0.0-max_new_tokens_8192-30samples-generations.jsonl"

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen3-14B/base_model_results-vllm/gsm8k_test_formatted/gsm8k_test_formatted-template_qwen3-temperature_0.0-max_new_tokens_8192-1319samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen3-14B/base_model_results-vllm/math_500/math_500-template_qwen3-temperature_0.0-max_new_tokens_8192-500samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen3-14B/base_model_results-vllm/mawps_formatted/mawps_formatted-template_qwen3-temperature_0.0-max_new_tokens_8192-238samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen3-14B/base_model_results-vllm/college_math_formatted/college_math_formatted-template_qwen3-temperature_0.0-max_new_tokens_8192-1200samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen3-14B/base_model_results-vllm/aime24_formatted/aime24_formatted-template_qwen3-temperature_0.0-max_new_tokens_8192-30samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen3-14B/base_model_results-vllm/aime25_formatted/aime25_formatted-template_qwen3-temperature_0.0-max_new_tokens_8192-30samples-generations.jsonl"

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/math_500/SAFE_Entropy_loss/lr_1e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/math_500/SAFE_Entropy_loss/lr_5e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/math_500/SAFE_Entropy_loss/lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/math_500/collapse_reg_entropy/lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/math_500/collapse_reg_entropy/lr_1e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/math_500-debug/naive_entropy/lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/math_500/collapse_reg_entropy/lr_1e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/base_model_results-hf-full_determinism/gsm8k_test_formatted/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/base_model_results-hf-full_determinism/math_500/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/gsm8k_test_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.2-3B-Instruct/base_model_results-vllm/minerva_formatted/minerva_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-272samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/base_model_results-hf/minerva_formatted/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-7B-Instruct/base_model_results-vllm/minerva_formatted/minerva_formatted-template_qwen-temperature_0.0-max_new_tokens_8192-272samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/base_model_results-vllm/minerva_formatted/minerva_formatted-template_None-temperature_0.0-max_new_tokens_8192-272samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen3-14B/base_model_results-vllm/minerva_formatted/minerva_formatted-template_None-temperature_0.0-max_new_tokens_8192-272samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/base_model_results-vllm/minerva_formatted/minerva_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-272samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-7B-Instruct/base_model_results-hf/college_math_formatted/generated_predictions.jsonl"

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent-observation3/gsm8k_test_formatted/naive_entropy-lowest/propotion_0.1-lr_7.5e-6-all_tokens-seed_42/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent-observation3/gsm8k_test_formatted/naive_entropy-lowest/propotion_0.2-lr_7.5e-6-all_tokens-seed_42/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent-observation3/gsm8k_test_formatted/naive_entropy-lowest/propotion_0.4-lr_7.5e-6-all_tokens-seed_42/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent-observation3/gsm8k_test_formatted/naive_entropy-lowest/propotion_0.6-lr_7.5e-6-all_tokens-seed_42/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent-observation3/gsm8k_test_formatted/naive_entropy-lowest/propotion_0.8-lr_7.5e-6-all_tokens-seed_42/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent-observation3/gsm8k_test_formatted/naive_entropy-lowest/propotion_1.0-lr_7.5e-6-all_tokens-seed_42/generated_predictions.jsonl",

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent-observation3/gsm8k_test_formatted/naive_entropy-highest/propotion_0.1-lr_7.5e-6-all_tokens-seed_42/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent-observation3/gsm8k_test_formatted/naive_entropy-highest/propotion_0.2-lr_7.5e-6-all_tokens-seed_42/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent-observation3/gsm8k_test_formatted/naive_entropy-highest/propotion_0.4-lr_7.5e-6-all_tokens-seed_42/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent-observation3/gsm8k_test_formatted/naive_entropy-highest/propotion_0.6-lr_7.5e-6-all_tokens-seed_42/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent-observation3/gsm8k_test_formatted/naive_entropy-highest/propotion_0.8-lr_7.5e-6-all_tokens-seed_42/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent-observation3/gsm8k_test_formatted/naive_entropy-highest/propotion_1.0-lr_7.5e-6-all_tokens-seed_42/generated_predictions.jsonl",
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/aime24_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/base_model_results-vllm/gsm8k_test_formatted/gsm8k_test_formatted-template_phi4-temperature_0.0-max_new_tokens_8192-1319samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/base_model_results-vllm/math_500/math_500-template_phi4-temperature_0.0-max_new_tokens_8192-500samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/base_model_results-vllm/college_math_formatted/college_math_formatted-template_phi4-temperature_0.0-max_new_tokens_8192-1200samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/base_model_results-vllm/aime24_formatted/aime24_formatted-template_phi4-temperature_0.0-max_new_tokens_8192-30samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/base_model_results-vllm/minerva_formatted/minerva_formatted-template_phi4-temperature_0.0-max_new_tokens_8192-272samples-generations.jsonl"

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/base_model_results-hf/minerva_formatted/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/base_model_results-hf/aime24_formatted/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/base_model_results-hf/gsm8k_test_formatted/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/base_model_results-hf/math_500/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/base_model_results-hf/college_math_formatted/generated_predictions.jsonl"
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/bnb_quantization-exp/quantitative_model_results-hf/gsm8k_test_formatted/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/bnb_quantization-exp/quantitative_model_results-hf/math_500/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/bnb_quantization-exp/quantitative_model_results-hf/college_math_formatted/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/bnb_quantization-exp/quantitative_model_results-hf/minerva_formatted/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/bnb_quantization-exp/quantitative_model_results-hf/aime24_formatted/generated_predictions.jsonl"
        
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/base_model_results-vllm/olympiad_formatted/olympiad_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-910samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-7B-Instruct/base_model_results-vllm/olympiad_formatted/olympiad_formatted-template_qwen-temperature_0.0-max_new_tokens_8192-910samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/base_model_results-vllm/olympiad_formatted/olympiad_formatted-template_None-temperature_0.0-max_new_tokens_8192-910samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/base_model_results-vllm/olympiad_formatted/olympiad_formatted-template_phi4-temperature_0.0-max_new_tokens_8192-910samples-generations.jsonl"

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-7B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-7B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-7B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_1e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-7B-Instruct/online_ttl/olympiad_formatted/lr_5e-6-all_tokens-threshold_3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-7B-Instruct/online_tent/olympiad_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_1e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/bnb_quantization-exp/quantitative_model_results-hf/olympiad_formatted/generated_predictions.jsonl"
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_1e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"

        # "/hujinwu/wyf/projects/zhangzian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-7B-Instruct/online_eata/olympiad_formatted/naive_entropy/lr_5e-6-all_tokens-threshold_0.4-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/bnb_quantization-exp/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/bnb_quantization-exp/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/bnb_quantization-exp/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_1e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_1e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_2.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_2.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/online_ttl/olympiad_formatted/lr_2.5e-6-all_tokens-threshold_3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/online_tent/olympiad_formatted/naive_entropy/lr_2.5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/online_tent/olympiad_formatted/dirichlet_entropy/lr_2.5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/online_eata/olympiad_formatted/lr_2.5e-6-all_tokens-threshold_0.4-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/bnb_quantization-exp/online_eata/olympiad_formatted/lr_1e-5-all_tokens-threshold_0.4-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/online_eata/olympiad_formatted/lr_2.5e-6-all_tokens-threshold_0.4-seed_42-v2/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/online_tent/olympiad_formatted/dirichlet_entropy/lr_2.5e-6-all_tokens-seed_42-v2/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/online_tent/olympiad_formatted/dirichlet_entropy/lr_2.5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/online_eata/olympiad_formatted/lr_2.5e-6-all_tokens-threshold_0.4-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/olympiad_formatted/dirichlet_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/olympiad_formatted/naive_entropy/lr_5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_2.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42-v2/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/bnb_quantization-exp/online_ttl/olympiad_formatted/lr_1e-5-all_tokens-threshold_3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/online_eata/olympiad_formatted/lr_2.5e-6-all_tokens-threshold_0.4-seed_42-v3/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/phi-4/online_tent/olympiad_formatted/dirichlet_entropy/lr_2.5e-6-all_tokens-seed_42-v3/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/bnb_quantization-exp/online_tent/olympiad_formatted/naive_entropy/lr_1e-5-tokens_6144-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/bnb_quantization-exp/online_tent/olympiad_formatted/dirichlet_entropy/lr_1e-5-tokens_6144-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_eata/olympiad_formatted/naive_entropy/lr_5e-6-all_tokens-threshold_0.4-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_ttl/olympiad_formatted/lr_5e-6-all_tokens-threshold_3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/online_tent/olympiad_formatted/naive_entropy/lr_7.5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/online_tent/olympiad_formatted/dirichlet_entropy/lr_7.5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/online_eata/olympiad_formatted/naive_entropy/lr_7.5e-6-all_tokens-threshold_0.4-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/online_ttl/olympiad_formatted/lr_7.5e-6-all_tokens-threshold_3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/gsm8k_test_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_math500/math_500-template_llama3-temperature_0.0-max_new_tokens_8192-500samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/gsm8k_test_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_college_math_formatted/college_math_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-1200samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/gsm8k_test_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_aime24_formatted/aime24_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-30samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/gsm8k_test_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_minerva_formatted/minerva_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-272samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/gsm8k_test_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_olympiad_formatted/olympiad_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-910samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/math_500/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_gsm8k_test_formatted/gsm8k_test_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-1319samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/math_500/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_aime24_formatted/aime24_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-30samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/math_500/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_college_math_formatted/college_math_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-1200samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/math_500/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_minerva_formatted/minerva_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-272samples-generations.jsonl"

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/college_math_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_gsm8k_test_formatted/gsm8k_test_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-1319samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/college_math_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_math_500/math_500-template_llama3-temperature_0.0-max_new_tokens_8192-500samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/math_500/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_olympiad_formatted/olympiad_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-910samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/college_math_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_aime24_formatted/aime24_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-30samples-generations.jsonl"
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/college_math_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_minerva_formatted/minerva_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-272samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/aime24_formatted/SAFE_Entropy_loss_detach_v2/lr_1.25e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_gsm8k_test_formatted/gsm8k_test_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-1319samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/college_math_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_olympiad_formatted/olympiad_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-910samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/aime24_formatted/SAFE_Entropy_loss_detach_v2/lr_1.25e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_math_500/math_500-template_llama3-temperature_0.0-max_new_tokens_8192-500samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/aime24_formatted/SAFE_Entropy_loss_detach_v2/lr_1.25e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_college_math_formatted/college_math_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-1200samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/aime24_formatted/SAFE_Entropy_loss_detach_v2/lr_1.25e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_minerva_formatted/minerva_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-272samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/minerva_formatted/SAFE_Entropy_loss_detach_v2/lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42-v2/to_gsm8k_test_formatted/gsm8k_test_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-1319samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/minerva_formatted/SAFE_Entropy_loss_detach_v2/lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42-v2/to_math_500/math_500-template_llama3-temperature_0.0-max_new_tokens_8192-500samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/minerva_formatted/SAFE_Entropy_loss_detach_v2/lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42-v2/to_aime24_formatted/aime24_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-30samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/minerva_formatted/SAFE_Entropy_loss_detach_v2/lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42-v2/to_college_math_formatted/college_math_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-1200samples-generations.jsonl"
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_gsm8k_test_formatted/gsm8k_test_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-1319samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/aime24_formatted/SAFE_Entropy_loss_detach_v2/lr_1.25e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_olympiad_formatted/olympiad_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-910samples-generations.jsonl"
    
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_aime24_formatted/aime24_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-30samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_math_500/math_500-template_llama3-temperature_0.0-max_new_tokens_8192-500samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_college_math_formatted/college_math_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-1200samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/to_minerva_formatted/minerva_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-272samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/minerva_formatted/SAFE_Entropy_loss_detach_v2/lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42-v2/to_olympiad_formatted/olympiad_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-910samples-generations.jsonl"
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/DeepSeek-R1-Distill-Qwen-7B/online_ttl/olympiad_formatted/lr_7.5e-6-all_tokens-threshold_3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-32B-Instruct/base_model_results-vllm/gsm8k_test_formatted/gsm8k_test_formatted-template_qwen-temperature_0.0-max_new_tokens_8192-1319samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-32B-Instruct/base_model_results-vllm/math_500/math_500-template_qwen-temperature_0.0-max_new_tokens_8192-500samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-32B-Instruct/base_model_results-vllm/college_math_formatted/college_math_formatted-template_qwen-temperature_0.0-max_new_tokens_8192-1200samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-32B-Instruct/base_model_results-vllm/aime24_formatted/aime24_formatted-template_qwen-temperature_0.0-max_new_tokens_8192-30samples-generations.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-32B-Instruct/base_model_results-vllm/minerva_formatted/minerva_formatted-template_qwen-temperature_0.0-max_new_tokens_8192-272samples-generations.jsonl"

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-32B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-32B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_1e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-32B-Instruct/base_model_results-vllm/olympiad_formatted/olympiad_formatted-template_qwen-temperature_0.0-max_new_tokens_8192-910samples-generations.jsonl"

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-32B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_2.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl",
        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-32B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"

        # "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Qwen2.5-32B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_2.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        #"/hujinwu/bly/OD-TTL/saves/output2/Qwen2.5-14B/tent/olympiad_formatted/1.0e-6/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
        "/hujinwu/bly/OD-TTL/saves/rebuttal/mixed_stream/Llama-3.1-8B/base/math_500_0.20_0.3/generated_predictions.jsonl"
    ]


    
    results = []
    for path in paths:
        acc, num_samples = get_acc(source=path, dataset_type="", verbose=False)  
        results.append(acc)
        print(f"总共 {num_samples} 个样本。准确率为：{acc}")

        # acc_list = get_acc_per_interval(path=path, log_to_file=True)
        # print(acc_list)
        # print(f"准确率为：{acc_list[-1]}")
    