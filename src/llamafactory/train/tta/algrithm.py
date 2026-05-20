import numpy as np
from collections import deque
import torch
from typing import Union, List, Dict
import math
import torch.nn.functional as F
from typing import Tuple

class PerTokenDynamicFutureGainSelector:
    """
    V4: 基于未来增益的词元选择器，具有逐词元动态窗口大小，并筛除低熵词元。
    
    此版本逻辑:
    1. 计算每个词元 t 的局部熵波动性 local_std(t)。
    2. 根据 local_std(t) 计算动态窗口 N(t)。
    3. 使用前缀和数组高效计算 Future_Gain(t) = Avg(t-N(t)..t-1) - Avg(t+1..t+N(t))。
    4. 先根据增益进行 Top-K/P 筛选，选出一批候选词元。
    5. **再从这批候选词元中，筛除自身熵值过低（接近于0）的词元。**
    """
    def __init__(self, local_window_size=6, gamma=1, n_min=2, n_max=10, top_p=0.2, top_k=None, entropy_threshold=1e-3):
        """
        Args:
            local_window_size (int): 计算局部熵波动性的邻域大小 M。
            gamma (float): 调节波动性与窗口大小关系的系数。
            n_min (int): 最小窗口大小。
            n_max (int): 最大窗口大小。
            top_p (float): 选择排名前 p% 的词元。 
            top_k (int, optional): 选择固定 K 个词元。
            entropy_threshold (float): 用于在Top-K/P选择后，筛除低熵词元的阈值。
        """
        self.local_window_size = local_window_size
        self.gamma = gamma
        self.n_min = n_min
        self.n_max = n_max
        self.top_p = top_p
        self.top_k = top_k
        self.entropy_threshold = entropy_threshold

    @torch.no_grad()
    def __call__(self, per_token_entropy, attention_mask, prompt_len):
        bs, seq_len = per_token_entropy.shape
        device = per_token_entropy.device
        
        # 准备工作
        total_len = attention_mask.sum(dim=1)
        
        # 初始化最终的增益、掩码和诊断信息
        future_gains = torch.full_like(per_token_entropy, -float('inf'))
        final_mask = torch.zeros_like(per_token_entropy, dtype=torch.bool)
        all_dynamic_n_t = torch.zeros_like(per_token_entropy, dtype=torch.long)

        # === 步骤 1-3: 逐样本计算未来增益 ===
        for i in range(bs):
            total_len_i = total_len[i].item()
            if total_len_i == 0: continue
            
            sample_entropy = per_token_entropy[i, :total_len_i]

            # 1. 计算局部熵波动性
            M = self.local_window_size
            padded_entropy = F.pad(sample_entropy.view(1, 1, -1), ((M-1)//2, M//2), 'replicate')
            windows = padded_entropy.unfold(2, M, 1)
            local_std = windows.std(dim=-1).squeeze()

            # 2. 计算每个词元 t 的动态窗口大小 N(t)
            dynamic_n_t = self.gamma / (local_std + 1e-6)
            dynamic_n_t = dynamic_n_t.round().long()
            dynamic_n_t = torch.clamp(dynamic_n_t, self.n_min, self.n_max)
            all_dynamic_n_t[i, :total_len_i] = dynamic_n_t

            # 3. 使用前缀和数组高效计算未来增益
            prefix_sum = torch.cumsum(sample_entropy, dim=0)
            prefix_sum = F.pad(prefix_sum, (1, 0))

            t_indices = torch.arange(total_len_i, device=device)
            N_t = dynamic_n_t

            past_start_idx = torch.clamp(t_indices - N_t, min=0)
            past_end_idx = t_indices
            past_sum = prefix_sum[past_end_idx] - prefix_sum[past_start_idx]
            past_avg = past_sum / N_t

            future_start_idx = torch.clamp(t_indices + 1, max=total_len_i)
            future_end_idx = torch.clamp(t_indices + N_t + 1, max=total_len_i)
            future_sum = prefix_sum[future_end_idx] - prefix_sum[future_start_idx]
            future_avg = future_sum / N_t
            
            sample_gain = past_avg - future_avg
            
            # 定义计算增益的有效范围 (此时不考虑熵阈值)
            prompt_len_i = prompt_len[i].item()
            is_in_generation = (t_indices >= prompt_len_i) & (t_indices < total_len_i)
            has_enough_history = (t_indices >= N_t)
            has_enough_future = (t_indices <= total_len_i - N_t - 1)
            valid_gain_mask = is_in_generation & has_enough_history & has_enough_future
            
            sample_gain[~valid_gain_mask] = -float('inf')
            future_gains[i, :total_len_i] = sample_gain

        # === 步骤 4-5: 先 Top-K/P 筛选，再用熵阈值过滤 ===
        for i in range(bs):
            # 找到所有具有有效增益的词元
            valid_gains_mask = future_gains[i] > -float('inf')
            valid_gains = future_gains[i][valid_gains_mask]
            
            if valid_gains.numel() == 0:
                continue
            
            # 步骤 4: 根据增益进行 Top-K 或 Top-P 筛选
            if self.top_k is not None:
                num_to_select = min(self.top_k, valid_gains.numel())
            else:
                num_to_select = max(1, int(valid_gains.numel() * self.top_p))

            if num_to_select == 0:
                continue
                
            _, top_indices_in_valid = torch.topk(valid_gains, k=num_to_select)
            
            # 获取这些候选词元在原始序列中的索引
            original_indices = torch.where(valid_gains_mask)[0]
            top_p_selected_indices = original_indices[top_indices_in_valid]

            # 步骤 5: 从候选词元中，筛除熵值过低的
            if top_p_selected_indices.numel() > 0:
                # 获取候选词元的熵值
                selected_entropies = per_token_entropy[i, top_p_selected_indices]
                
                # 创建一个掩码，只保留熵值高于阈值的词元
                high_entropy_mask = selected_entropies > self.entropy_threshold
                
                # 应用掩码，得到最终被选中的词元索引
                final_selected_indices = top_p_selected_indices[high_entropy_mask]
                
                # 更新最终的掩码
                final_mask[i, final_selected_indices] = True
        
        diagnostics = {
            "dynamic_n_t": all_dynamic_n_t,
            "future_gains": future_gains,
        }
        
        return final_mask, diagnostics


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps
