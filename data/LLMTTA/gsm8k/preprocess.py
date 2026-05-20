from datasets import load_dataset
import json

dataset = load_dataset('/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory/data/tta/gsm8k', data_files='test_clean.json')['train']
houzhui = " Let's think step by step and output the final answer within \\boxed{}."
save_path = '/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory/data/tta/gsm8k/test_formatted_v2.jsonl'
with open(save_path, 'w', encoding='utf-8') as f:
    for sample in dataset:
        data = {
            **sample,
            "instruction": sample['question'] + houzhui,
            "input": '',
            "output": sample['answer'].split('#### ')[-1]
        }
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
        f.flush()
