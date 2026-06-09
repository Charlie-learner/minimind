from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    对话前处理：以一定概率随机插入 system 消息。

    特点：
    - 只有当首条消息不是 system 角色时才可能插入。
    - add_system_ratio 控制插入概率（默认 20%），引入随机性可提升模型
      对有/无 system prompt 两种情况的泛化能力。
    - system 内容从预定义的中英文 prompt 池中随机抽取，覆盖不同表达风格。
    """
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
            ] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    对话后处理：清理模板渲染后多余的空 <think> 块。

    特点：
    - 针对带 CoT（chain-of-thought）格式的模型，apply_chat_template 有时会
      渲染出 "<think>\n\n</think>\n\n" 这样的空思考块占位符。
    - 大部分情况下（概率 1 - empty_think_ratio = 95%）直接删除该空块，
      防止模型学到"无意义思考"的坏习惯。
    - 保留少量空思考块（empty_think_ratio = 5%），让模型也能处理该边界情况。
    """
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        and random.random() > empty_think_ratio
    ):
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content




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
        self.samples = load_dataset("json", data_files=data_path, split="train")

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




# ──────────────────────────────────────────────────────────────────────────────
# 2. SFTDataset —— 有监督微调（Supervised Fine-Tuning）数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：让模型学会"只预测 assistant 回复"，忽略 user/system 输入
# 数据格式：{"conversations": [{"role": "user"/"assistant"/"system", "content": "..."}]}
# 训练特点：
#   - 通过 generate_labels 扫描 bos_id（assistant 回复起始标记）定位每段回复，仅将 assistant 回复的 token 位置设为有效 label，其余全部为 -100。
#   - 这样做的意义：让 loss 只反映模型对"正确回答"的拟合，不浪费梯度在用户输入的复现上（用户输入只作为 context，不是预测目标）。
#   - 支持 function calling：若 system 消息携带 "functions" 字段，会透传给 apply_chat_template，生成带工具描述的提示词。
#   - 与 PretrainDataset 的关键区别：标签是"稀疏"的，只有 assistant 部分非 -100。
# ──────────────────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({"conversations":[{"role": Value("string"), "content": Value("string"), "reasoning_content": Value("string"), "tools": Value("string"), "tool_calls": Value("string")}]})
        self.samples = load_dataset("json", data_files=jsonl_path, split="train", features=features)
        # tokenizer() 输出input_ids, attention_mask, token_type_ids(可选输出), ...
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)
    
    def create_chat_prompt(self, conversations):
        """
        将多轮对话转换为模型输入的字符串。

        特点：
        - 复制原始 conversations，防止修改原始数据。
        - 检测 system 消息中是否携带 functions 字段（function calling 场景），
          若有则透传给 apply_chat_template，生成标准 tool-use 格式的提示词。
        - add_generation_prompt=False：不在末尾追加"请模型续写"的 prompt，
          因为训练时需要完整的 input+output 序列，而非开放续写。
        """
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)     # 深拷贝, 安全隔离，防止修改原始数据
            if message.get("role") == "system" and message.get("tools"):
                tools = json.load(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.load(message["tool_calls"])
            messages.append(message)

        # tokenize=False: 不进行 tokenization; add_generation_prompt=False: 不在末尾追加"请模型续写"的 prompt;
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
    
    def generate_labels(self, input_ids):
        """
        生成 SFT 训练所需的稀疏标签序列。
        
        算法逻辑（滑动窗口扫描）：
        1. 初始化全 -100 的 labels，默认所有位置不计算 loss。
        2. 逐位扫描 input_ids，检测是否匹配 bos_id（assistant 回复起始）。
        3. 匹配到 bos_id 后，向后扫描直到找到 eos_id（回复结束）。
        4. 将 [start, end+len(eos_id)) 区间内的 label 设为对应的 input_ids 值，
           即这段 assistant 的回复参与 loss 计算。
        5. EOS token 本身也计入 label，让模型学会何时停止生成。
        6. 跳过已处理区间，继续扫描下一段 assistant 回复（支持多轮对话）。
        """
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i: i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        end += len(self.eos_id)
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos.id), self.max_length)):
                    labels[j] = input_ids[j]
                
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1

        return labels
                               
    def __getitem__(self, index):
        sample = self.samples[index]

        # 1. 随机决定是否插入 system prompt (数据增强)
        conversations = pre_processing_chat(sample["conversations"])
        # 2. 用 chat template 渲染完整对话字符串
        prompt = self.create_chat_prompt(conversations)
        # 3. 清理可能出现的空 <think>\n</think> 块
        prompt = post_processing_chat(prompt)
        # 4. tokenize 并截断到 max_length, 不足则右侧 PAD 补齐
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 5. 生成稀疏标签，只有 assistant 回复参与 loss 计算
        labels = self.generate_labels(input_ids)

        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)



