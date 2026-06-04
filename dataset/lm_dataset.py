from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ──────────────────────────────────────────────────────────────────────────────
# 1. PretrainDataset —— 自回归预训练数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：Next-Token Prediction（下一个 token 预测）
# 数据格式：{"text": "一段原始文本"}
# 训练特点：
#   - 模型对整段文本的每个位置都进行预测，没有"只学回复"的区分。
#   - 使用 BOS/EOS 标记文本边界，让模型学会文本的起止。
#   - PAD token 对应的 label 置 -100，不参与 loss 计算，节省无效梯度。
#   - labels 直接 clone 自 input_ids（即 X 和 Y 错位一格：Y[t] = X[t+1]）。
# ──────────────────────────────────────────────────────────────────────────────
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 Hugging Face datasets 的惰性加载，避免一次性读入大文件
        # self.samples：加载出来的结果。它不是普通的 Python 列表，而是一个虚拟的内存映射对象。
        # 即使你的 JSON 文件有 100GB 大，这行代码也能瞬间跑完，因为它不会把数据全读进内存，而是用到哪一行才去硬盘读哪一行，极其省内存。
        self.samples = load_dataset("json", data_file=data_path, split="train")

    def __len__(self):
        return len(self.samples)
    
    #
    def __getitem__(self, index):
        sample = self.samples[index]     # sample 是一个字典，结构类似于 {"text": "今天天气真好"}

        # 1. Tonkenize 原始文本，留出首尾各 1 个 token 的位置给 BOS/EOS
        tokens = self.tokenizer(
            str(sample["text"]),   
            add_special_tokens=False, 
            max_length=self.max_length - 2, # 预留 BOS/EOS 的位置
            truncation=True,
            ).input_ids
        
        # 2.拼接 BOS + token 序列 + EOS，构成完整序列
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # 3. 用 PAD 补齐序列长度到 max_length，保留 batch 内等长
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 4. labels 与 input_ids 相同，除了 PAD 位置置 -100
        #    CrossEntropyLoss 会自动忽略 -100，不计入 loss
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # 5. 返回 attention_mask, 使其屏蔽 padding token
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return input_ids, labels, attention_mask




