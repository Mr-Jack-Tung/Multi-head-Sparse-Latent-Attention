# pip install transformers==4.53.1 torch==2.2.2 numpy==1.26.4 datasets
# pip install 'transformers[torch]'
# Note: By downgrading to NumPy 1.26.4, we provided a version that PyTorch 2.2.2 can properly interact with Python 3.10~3.12
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2PreTrainedModel, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutput
import math

class MSLA(nn.Module):
    """
    Multi-head Sparse Latent Attention (MSLA)

    Args:
        embed_dim: hidden size
        num_heads: number of attention heads
        num_latents: number of learnable latent keys
        k_top: number of latent keys to keep (sparse selection)
    """
    def __init__(self, embed_dim, num_heads, num_latents=64, k_top=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.k_top = k_top

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Learnable latent keys (num_latents x embed_dim)
        self.latent_keys = nn.Parameter(torch.randn(num_latents, embed_dim)* 0.1)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, embed_dim)

        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        B, T, D = hidden_states.size()

        # Linear projections
        Q = self.q_proj(hidden_states)  # (B, T, D)
        K = self.k_proj(hidden_states)  # (B, T, D)
        V = self.v_proj(hidden_states)  # (B, T, D)

        # Reshape for multi-head
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D_head)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D_head)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D_head)

        # Project latent keys to per-head representation
        latent = self.latent_keys.view(self.num_latents, self.num_heads, self.head_dim)  # (L, H, D_head)

        # Compute attention logits: Q x latent_keys.T
        attn_logits = torch.einsum('bhtd,lhd->bhtl', Q, latent)  # (B, H, T, L)
        attn_logits = attn_logits / math.sqrt(self.head_dim)

        # Top-k selection on latent dimension
        topk_logits, topk_indices = torch.topk(attn_logits, self.k_top, dim=-1)  # (B,H,T,k)

        # Softmax over k selected latents
        attn_weights = F.softmax(topk_logits, dim=-1)  # (B,H,T,k)

        # Gather latent keys corresponding to topk_indices
        # latent: (L,H,D_head)
        # We need: (B,H,T,k,D_head)

        # First, reshape for gather
        latent_exp = latent.permute(1,0,2).unsqueeze(0).unsqueeze(2)  # (1,H,1,L,D_head)
        latent_exp = latent_exp.expand(B, -1, T, -1, -1)  # (B,H,T,L,D_head)

        # topk_indices: (B,H,T,k) -> need to expand last dim
        topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)  # (B,H,T,k,D_head)

        # Gather
        selected_latents = torch.gather(latent_exp, dim=3, index=topk_indices_exp)  # (B,H,T,k,D_head)

        # Weighted sum over k latent keys (you can design other mixing strategies)
        # attn_weights: (B,H,T,k) -> unsqueeze(-1)
        weighted_latents = (attn_weights.unsqueeze(-1) * selected_latents).sum(dim=3)  # (B,H,T,D_head)

        # Combine latent-aware output with V
        # Here we do a residual mix: latent + V
        attn_output = weighted_latents + V  # (B,H,T,D_head)

        # Merge heads
        attn_output = attn_output.transpose(1,2).contiguous().view(B,T,D)  # (B,T,D)

        return self.out_proj(attn_output)

def patch_gpt2_with_msla(model, num_latents, k_top):
    print("[🔧] Patching GPT2 Attention -> MSLA ...")
    for i, block in enumerate(model.transformer.h):
        embed_dim = block.attn.c_attn.weight.shape[0]
        num_heads = block.attn.num_heads

        # Tách QKV
        W = block.attn.c_attn.weight
        B = block.attn.c_attn.bias

        q_weight = W[:, :embed_dim].T.contiguous()
        k_weight = W[:, embed_dim:2*embed_dim].T.contiguous()
        v_weight = W[:, 2*embed_dim:].T.contiguous()

        q_bias = B[:embed_dim]
        k_bias = B[embed_dim:2*embed_dim]
        v_bias = B[2*embed_dim:]

        # Khởi tạo MSLA - Initialize MSLA
        # embed_dim: 768, num_heads: 12, num_latents: 64, k_top: 4
        msla = MSLA(embed_dim, num_heads, num_latents, k_top)

        with torch.no_grad():
            msla.q_proj.weight.copy_(q_weight)
            msla.k_proj.weight.copy_(k_weight)
            msla.v_proj.weight.copy_(v_weight)

            msla.q_proj.bias.copy_(q_bias)
            msla.k_proj.bias.copy_(k_bias)
            msla.v_proj.bias.copy_(v_bias)

        # Disable c_attn
        block.attn.c_attn = torch.nn.Identity()

        # Attach MSLA
        block.attn.msla = msla

        def new_forward(self, hidden_states, *args, **kwargs):
            attn_output = self.msla(hidden_states)
            attn_output = self.c_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)
            return attn_output, None

        block.attn.forward = new_forward.__get__(block.attn, type(block.attn))
    print("[✅] MSLA Attention successfully patched.")

# Lớp wrapper
from transformers import GPT2LMHeadModel, GPT2Config
import torch
import os

def from_pretrained_with_msla(pretrained_path, num_latents=64, k_top=4, device="cpu"):
    """
    Load GPT2LMHeadModel + MSLA + all trained weights.
    """
    print("[🔧] Loading GPT2 config...")
    config = GPT2Config.from_pretrained(pretrained_path)
    
    print("[🔧] Instantiating GPT2 model...")
    model = GPT2LMHeadModel(config)
    
    print("[💾] Loading base state_dict...")
    # support safetensors or pytorch bin
    if os.path.exists(os.path.join(pretrained_path, "model.safetensors")):
        from safetensors.torch import load_file as safe_load
        state_dict = safe_load(os.path.join(pretrained_path, "model.safetensors"))
    else:
        state_dict = torch.load(os.path.join(pretrained_path, "pytorch_model.bin"), map_location=device)

    print("[🔧] Loading base weights...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[✅] Base weights loaded. Missing: {len(missing)} keys, Unexpected: {len(unexpected)} keys")

    # Patch MSLA
    print("[🔧] Patching MSLA into GPT2...")
    for i, block in enumerate(model.transformer.h):
        embed_dim = block.attn.c_attn.weight.shape[0]
        num_heads = block.attn.num_heads
        
        # Tách QKV weight (dự phòng nếu chưa có weight MSLA)
        W = block.attn.c_attn.weight
        B = block.attn.c_attn.bias

        q_weight = W[:, :embed_dim].T.contiguous()
        k_weight = W[:, embed_dim:2*embed_dim].T.contiguous()
        v_weight = W[:, 2*embed_dim:].T.contiguous()

        q_bias = B[:embed_dim]
        k_bias = B[embed_dim:2*embed_dim]
        v_bias = B[2*embed_dim:]

        msla = MSLA(embed_dim, num_heads, num_latents, k_top)

        # Copy QKV dự phòng
        with torch.no_grad():
            msla.q_proj.weight.copy_(q_weight)
            msla.k_proj.weight.copy_(k_weight)
            msla.v_proj.weight.copy_(v_weight)
            msla.q_proj.bias.copy_(q_bias)
            msla.k_proj.bias.copy_(k_bias)
            msla.v_proj.bias.copy_(v_bias)

        # Thay c_attn bằng Identity
        block.attn.c_attn = torch.nn.Identity()

        # Attach MSLA
        block.attn.msla = msla

        # Load lại weight MSLA từ state_dict
        prefix = f"transformer.h.{i}.attn.msla."
        msla_keys = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

        msla_missing, msla_unexpected = msla.load_state_dict(msla_keys, strict=False)
        
        if msla_missing or msla_unexpected:
            print(f"[✅] MSLA block {i} weights loaded. Missing: {len(msla_missing)} keys, Unexpected: {len(msla_unexpected)} keys")

        # Patch forward
        def new_forward(self, hidden_states, *args, **kwargs):
            attn_output = self.msla(hidden_states)
            attn_output = self.c_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)
            return attn_output, None
        
        block.attn.forward = new_forward.__get__(block.attn, type(block.attn))
    
    print("[✅] All MSLA blocks patched and weights loaded.")
    return model
