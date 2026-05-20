
#################################################
# AAAI26
#################################################

# model=/hujinwu/LLM_Assemble/pretrain_model/phi-4
# $adapter_name_or_path/predict \
# --adapter_name_or_path $adapter_name_or_path \
# model=/hujinwu/LLM_Assemble/pretrain_model/Meta-Llama-3-8B-Instruct
# model=/hujinwu/LLM_Assemble/pretrain_model/Llama-3.2-3B-Instruct
# model=/hujinwu/LLM_Assemble/pretrain_model/Llama-2-13b-chat-hf

# model=/hujinwu/LLM_Assemble/pretrain_model/Qwen2.5-7B-Instruct
# model=/hujinwu/LLM_Assemble/pretrain_model/DeepSeek-R1-Distill-Qwen-7B

# model=/hujinwu/LLM_Assemble/pretrain_model/Llama-2-13b-chat-hf
# model=/hujinwu/LLM_Assemble/pretrain_model/Llama-3.1-8B-Instruct

# model=/hujinwu/LLM_Assemble/pretrain_model/Qwen3-14B

# adapter_name_or_path=/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory/saves/after_rebuttal/qwen2.5-7b/offline_tent_come/instruction_wild_5k/lr_5e-5-max_new_tokens_80-seed_42
# adapter_name_or_path=/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory/saves/llama3-8b/lora/select_sample_0108/geosignal/threshold_3-lamb_0.1-lr_5e-5
# adapter_name_or_path=/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory/saves/after_rebuttal/llama2-13b/offline_tent-newImp/gsm8k_5k/lr_1e-6-max_new_tokens_80-seed_42
    # --template deepseek \

MODEL_PATH_ROOT=/hujinwu/LLM_Assemble/pretrain_model
# BACKBONE=Qwen2.5-7B-Instruct
# BACKBONE=Llama-3.1-8B-Instruct
# BACKBONE=Llama-2-13b-chat-hf
# BACKBONE=Qwen3-14B
BACKBONE=phi-4
# BACKBONE=DeepSeek-R1-Distill-Qwen-7B
BACKBONE=Qwen2.5-32B-Instruct
# TEMPLATE=qwen3
# TEMPLATE=qwen
# TEMPLATE=none
# TEMPLATE=qwen
TEMPLATE=llama3
MODEL=/hujinwu/pre-models/Meta-Llama-3.1-8B-Instructls
DATASET=gsm8k_noisy  # gsm8k_test_formatted, gsm8k_test_add_suffix, math_500, mawps_formatted, college_math_formatted, aime24_formatted, aime25_formatted, minerva_formatted, olympiad_formatted

 python scripts/vllm_infer.py --model_name_or_path $MODEL \
    --template $TEMPLATE \
    --dataset $DATASET \
    --cutoff_len 1024 \
    --temperature 0.0 \
    --max_new_tokens 8192 \
    --output_dir /hujinwu/bly/OD-TTL/saves/rebuttal/mixed_stream/Llama-3.1-8B/base/$DATASET \
    --vllm_config "{gpu_memory_utilization: 0.95, tensor_parallel_size: 1}" \

# gsm8k_test_formatted, gsm8k_test_add_suffix, math_500, college_math_formatted, aime24_formatted, aime25_formatted, minerva_formatted, olympiad_formatted
# geosignal_5k, agriculture_5k, gen_med_gpt_5k, wealth_5k

# model_path=/hujinwu/LLM_Assemble/pretrain_model/Llama-3.1-8B-Instruct
# # adapter_path=/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/gsm8k_test_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42
# adapter_path=/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/math_500/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42
# # adapter_path=/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/college_math_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42
# # adapter_path=/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/aime24_formatted/SAFE_Entropy_loss_detach_v2/lr_1.25e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42
# # adapter_path=/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/minerva_formatted/SAFE_Entropy_loss_detach_v2/lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42-v2
# # adapter_path=/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/olympiad_formatted/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42

# python scripts/vllm_infer.py --model_name_or_path $model_path \
#     --adapter_name_or_path $adapter_path \
#     --template llama3 \
#     --dataset gen_med_gpt_5k \
#     --cutoff_len 1024 \
#     --temperature 0.0 \
#     --max_new_tokens 512 \
#     --output_dir $adapter_path/to_gen_med_gpt_5k \
#     --vllm_config "{gpu_memory_utilization: 0.95, tensor_parallel_size: 1}" \