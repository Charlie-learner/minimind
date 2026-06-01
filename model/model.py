from transformers import PretrainedConfig

#################################################################
# MiniMind Config
#################################################################
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


#################################################################
# MiniMind Model
#################################################################
import torch
import torch.nn as nn

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