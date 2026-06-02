import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

#################################################################
# MiniMind Config
#################################################################
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
#################################################################
# MiniMind Model
#################################################################
class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        def norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdims=True) + self.eps)  
        
        # 在训练或推理现代大语言模型时，为了节省显存和加速计算，输入的 x 通常是低精度数据类型，比如 float16 或 bfloat16。
        # float16 的最大表示范围只有 65504，在pow(2)运算中很容易出现溢出，导致结果为 inf 或 nan，且在求和/求平均时，低精度类型会带来累加误差（累加误差），使得结果不稳定。
        # 因此，在计算 RMSNorm 时，先将输入 x 转换为 float32 进行计算，确保数值稳定性和准确性。最后再将结果转换回输入 x 的数据类型，以保持与模型其他部分的一致性。
        # 此为“高精度计算，低精度存储/传输”
        def forward(self, x):
            return self.weight * self.norm(x.float()).type_as(x) 


# precompute_freqs_cis 函数：预计算旋转频率矩阵
# dim：注意力头的特征维度（Head_dim)，end: 预计算的最大序列长度（上下文窗口上限）， rope_base：RoPE 运算的底数基准值(base)(科学计数法在python中原生就是float), rope_scaling: 包含 YaRN 算法超参数的字典，若为 None 则执行标准 RoPE
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    # 1. 初始化标准 RoPE 频率。
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arrange(0, dim, 2)[:(dim // 2)].float() / dim)), 1.0

    if rope_scaling is not None:    # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32),
            rope_scaling.get("beta_slow", 1),
            rope_scaling.get("attention_factor", 1.0),
        )

        if end > orig_max:
            # 3. 通过在训练长度内旋转的周期数来计算出维度索引 i
            inv_dim = lambda b: (dim * math.log(orig_max / (2 * math.pi * b))) / (2 * math.log(rope_base))
            # 4. 计算出高频区和低频区的维度边界索引 low 和 high
            # low 是高频边界索引，high 是低频边界索引。维度索引 i 小于 low 的部分属于高频区，不进行缩放；维度索引 i 大于 high 的部分属于低频区，进行全量缩放；维度索引 i 在 low 和 high 之间的部分属于过渡区，进行线性过渡插值缩放。
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)

            # 5. 构造线性 ramp，计算出每个维度的缩放因子，并应用到频率上
            # 维度索引 i 小于 low 的部分，ramp 为 0，缩放因子为 1，不缩放频率；维度索引 i 大于 high 的部分，ramp 为 1，缩放因子为 1/factor，频率缩小 factor 倍；维度索引 i 在 low 和 high 之间的部分，ramp 在 0 和 1 之间线性变化，缩放因子在 1 和 1/factor 之间线性变化，实现平滑过渡。
            # torch.clamp 用于限制 ramp 的值在 0 和 1 之间，避免过渡区之外的维度出现异常缩放。
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)

            # 6. 最后将计算出的缩放因子应用到频率上，得到最终的 YaRN 频率矩阵。高频区的频率保持不变，低频区的频率缩小 factor 倍，过渡区的频率根据 ramp 线性插值缩放。
            freqs = freqs * (1.0 - ramp + ramp / factor)

    # 7. 根据预计算的最大序列长度，生成位置索引向量 pos_idx，范围从 0 到 end-1。
    pos_idx = torch.arange(end, device=freqs.device)

    # 8. 将频率矩阵和位置索引向量进行外积，得到一个形状为 (end, dim // 2) 的频率位置矩阵 freqs
    freqs = torch.outer(pos_idx, freqs)

    # 9. 根据频率位置矩阵，计算出对应的余弦和正弦矩阵，并进行拼接和注意力温度补偿，得到最终的频率矩阵 freqs_cos 和 freqs_sin。
    # 拼接是为了适配旋转位置编码中交替使用余弦和正弦函数的方式，注意力温度补偿是为了应对长距离依赖时注意力分布变平缓的问题，让模型能够更好地聚焦于相关位置。
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim = -1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim = -1) * attn_factor

    return freqs_cos, freqs_sin


# apply_rotary_pos_emb 函数：将预计算的频率矩阵应用到查询向量 q 和键向量 k 上，得到带有旋转位置编码的嵌入向量 q_embed 和 k_embed。
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # 为了在高效的矩阵乘法中避免逐元素切片造成的算子碎片化，工程上常采用 rotate_half 变体：把长度为 d 的特征向量直接对半切开，将后半部分整体加负号移到前面，前半部分移到后面：
    # 论文的交错式：公式规定第 2i 维和第 2i+1 维（邻居）要一起转。
    # 代码的对半拆分式：公式规定第 i 维和第 i + d/2 维（远亲）要一起
    # 神经网络里的词向量维度（比如 $d=128$），在刚初始化的时候，第 0 维和第 1 维是没有先后顺序或特殊绑定关系的。维度之间的位置和语义，完全是由后面训练时的数学公式“强行赋予”的。
    # 对于大模型里的线性投影层 Wq 和 Wk 来说，它在训练时会“自适应”
    # 故此代码的对半拆分式在全局宏观计算上与交错式的两两旋转完全等价，但对 GPU 硬件计算极度友好。
    def rotate_half(x):
        return torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
    
    # .to(q.dtype)：因为频率计算强制在精度更高的 float32 下进行，算完位置编码后，需要用此操作将结果强制降采样回 $q$ 原本的半精度数据类型（如 float16 或 bfloat16），确保后续层计算的类型一致。
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)

    return q_embed, k_embed


def repeate_kv(x: torch.Tensor, num_repeats: int):
    # x: [batch_size, seq_len, num_heads, head_dim]
    bsz, seq_len, num_heads, head_dim = x.shape
    if num_repeats == 1:
        return x    

    return (x.unsqueeze(3).expand(-1, -1, -1, num_repeats, -1).reshape(bsz, seq_len, num_heads * num_repeats, head_dim))


class Attention(nn.Module):
    def __init__(self, config:MiniMindConfig):
        super().__init__()
        
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        
        assert config.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        
        self.n_local_heads = config.num_attention_heads  # 此处由于是单卡，所以没有除以 world_size，保持和 num_attention_heads 一致
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_repeats = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        
        self.is_casual = True
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias = False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias = False)
        
        self.q_norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.k_norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash_attention

    def forward(self, x: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache = False, attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        # 1. 线性投影得到查询、键、值向量，并调整形状以适配多头注意力机制
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        # 2. 计算 RoPE 位置编码并应用到查询向量和键向量上
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 3. kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        
        # 4. 重复键值向量以匹配查询向量的头数（如果 n_repeats > 1），并调整形状以适配注意力计算
        xq, xk, xv = (xq.transpose(1, 2), repeate_kv(xk, self.n_repeats).transpose(1, 2), repeate_kv(xv, self.n_repeats).transpose(1, 2))

        # 5. 计算注意力输出。根据条件选择使用 Flash Attention 或标准的缩放点积注意力计算。
        if self.flash and (seq_len > 1) and (not self.is_casual or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout = self.dropout if self.training else 0.0, is_causal = self.is_casual)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_casual:
                scores[..., -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device = scores.device), diagonal = 1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * float("-inf")
            output = self.attn_dropout(F.softmax(scores.float(), dim = -1).type_as(xv)) @ xv
        
        # 6. 将多头注意力输出重新组合并通过输出投影层，得到最终的注意力输出，并返回输出和新的 kv_cache。
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_local_heads * self.head_dim)
        output = self.residual_dropout(self.o_proj(output))
        
        return output, past_kv
    

class FeedForward(nn.Module):
    # 初始化、升维、降维、门控、激活函数、dropout
    def __init__(self, config:MiniMindConfig):
        super().__init__()

        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias = False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias = False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias = False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout == nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

# MiniMind Block 是一个 Transformer 块，包含一个自注意力层和一个前馈网络层。它还包括两个 RMSNorm 层，分别在输入和注意力输出后进行归一化。前馈网络可以选择使用 MoE（Mixture of Experts）版本，以增加模型容量和表达能力。
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()

        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MoEFeedForward(config)
        self.input_layernorm

    def forward(
            self, 
            hidden_states, 
            position_embeddings: Tuple[torch.Tensor, torch.Tensor], 
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
            use_cache=False, 
            attention_mask: Optional[torch.Tensor] = None
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), 
            position_embeddings = position_embeddings,
            past_key_value = past_key_value,
            use_cache = use_cache,
            attention_mask = attention_mask,
        )

        # 注意力输出与残差连接相加后，再经过一个前馈网络层和第二个残差连接，得到最终的块输出。
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value
        

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(i, config) for i in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        
        # 预计算 RoPE 频率矩阵，缓存并注册为非训练张量（自动对齐设备/精度），persistent = False 设为不持久化以防占用磁盘权重体积
        freq_cos, freq_sin = precompute_freqs_cis(config.head_dim, end = config.max_position_embeddings, rope_base = config.rope_theta, rope_scaling = config.rope_scaling)
        self.register_buffer("freq_cos", freq_cos, persistent = False)
        self.register_buffer("freq_sin", freq_sin, persistent = False)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None, 
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            **kwargs
    ):
        # input_ids: [batch_size, seq_len]
        bsz, seq_len = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None

        past_key_values = past_key_values if past_key_values is not None else [None] * len(self.layers)

        # 计算start_pos以确定当前输入序列在整体上下文中的起始位置，这对于正确应用 RoPE 位置编码非常重要。
        # start_pos 的计算方式是检查 past_key_values 中第一个元素（对应第一层）的键向量的形状，如果存在，则取其序列长度（shape[1]），否则默认为 0。这意味着如果有过去的键值对缓存，start_pos 将反映已经处理过的序列长度，从而确保新的输入序列能够正确地接续在之前的上下文之后进行位置编码。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # Embedding + dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # position_embeddings
        position_embeddings = (self.freq_cos[start_pos : start_pos + seq_len], self.freq_sin[start_pos : start_pos + seq_len])

        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states, 
                position_embeddings = position_embeddings,
                past_key_value = past_key_value,
                use_cache = use_cache,
                attention_mask = attention_mask,
            )
            presents.append(present)
            
        hidden_states = self.norm(hidden_states)

        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MoEFeedForward)], hidden_states.new_zeros(1).squeze())

        return hidden_states, presents, aux_loss

        
class MiniMindForCausalLM(PretrainedConfig, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig):
        super().__init__(config)
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)
        self.model.embed_tokens.weight = self.lm_head.weight if config.tie_word_embeddings else self.model.embed_tokens.weight

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            logits_to_keep: Union[int, torch.Tensor] = 0, 
            **kwargs
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        
        # 动态计算切片：推理时常设为1仅切出最后一个Token，训练时设为0保留全量文本
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 仅对切片后的特征执行线性投影，避免在推理阶段对历史Token做重复计算，极大地压缩显存与算力开销
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 计算交叉熵损失
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output