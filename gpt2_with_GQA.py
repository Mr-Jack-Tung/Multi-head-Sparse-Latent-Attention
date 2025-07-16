import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GQA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_key_value_groups=2):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_key_value_groups == 0, "num_heads must be divisible by num_key_value_groups"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_key_value_groups = num_key_value_groups
        self.head_dim = embed_dim // num_heads
        self.kv_heads = num_key_value_groups
        self.kv_dim = self.kv_heads * self.head_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.kv_dim)
        self.v_proj = nn.Linear(embed_dim, self.kv_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()

        # Linear projection
        q = self.q_proj(x)         # (B, T, C)
        k = self.k_proj(x)         # (B, T, kv_dim)
        v = self.v_proj(x)         # (B, T, kv_dim)

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)   # (B, G, T, D)
        v = v.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)   # (B, G, T, D)

        # Repeat k/v for each query head group
        expand_ratio = self.num_heads // self.kv_heads
        k = k.repeat_interleave(expand_ratio, dim=1)  # (B, H, T, D)
        v = v.repeat_interleave(expand_ratio, dim=1)  # (B, H, T, D)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5  # (B, H, T, T)

        if attention_mask is not None:
            attn_scores += attention_mask  # mask broadcast

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, D)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)

def patch_gpt2_with_gqa(model, num_key_value_groups=2):
    print("[🔧] Patching GPT2 Attention -> GQA ...")
    for i, block in enumerate(model.transformer.h):
        embed_dim = block.attn.embed_dim
        num_heads = block.attn.num_heads

        W = block.attn.c_attn.weight   # (768,2304)
        b = block.attn.c_attn.bias     # (2304,)

        # print("c_attn.weight shape:", W.shape)
        # print("c_attn.bias shape:", b.shape)

        # ✅ Slice chiều 1
        q_weight = W[:, :embed_dim]
        k_weight = W[:, embed_dim:2*embed_dim]
        v_weight = W[:, 2*embed_dim:]

        q_bias = b[:embed_dim]
        k_bias = b[embed_dim:2*embed_dim]
        v_bias = b[2*embed_dim:]

        gqa = GQA(embed_dim, num_heads, num_key_value_groups)
        kv_dim = gqa.kv_dim

        # if i == 0:
        #     print(f"Block {i}:")
        #     print(f"  q_weight: {q_weight.shape}")
        #     print(f"  k_weight: {k_weight.shape}")
        #     print(f"  v_weight: {v_weight.shape}")
        #     print(f"  kv_dim: {kv_dim}")

        with torch.no_grad():
            gqa.q_proj.weight.copy_(q_weight.T)
            gqa.q_proj.bias.copy_(q_bias)

            gqa.k_proj.weight.copy_(k_weight[:, :kv_dim].T)
            gqa.k_proj.bias.copy_(k_bias[:kv_dim])

            gqa.v_proj.weight.copy_(v_weight[:, :kv_dim].T)
            gqa.v_proj.bias.copy_(v_bias[:kv_dim])

            gqa.out_proj.weight.copy_(block.attn.c_proj.weight)
            gqa.out_proj.bias.copy_(block.attn.c_proj.bias)

        block.attn.gqa = gqa

        def new_forward(hidden_states, attention_mask=None, head_mask=None, **kwargs):
            output = block.attn.gqa(hidden_states, attention_mask)
            return output, None

        block.attn.forward = new_forward
        block.attn.c_attn = nn.Identity()

    print("[✅] GQA Attention successfully patched.")
