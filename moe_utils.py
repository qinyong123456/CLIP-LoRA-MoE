import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MoEMLP(nn.Module):
    """
    MoE replacement for CLIP ResidualAttentionBlock's MLP.
    - Experts: copies of the original MLP (c_fc -> gelu -> c_proj)
    - Router: linear gating over hidden features, soft mixture combine
    - Aux loss: load balancing via KL(mean_prob || uniform)
    """
    def __init__(self, base_mlp: nn.Sequential, d_model: int, num_experts: int):
        super().__init__()
        assert isinstance(base_mlp, nn.Sequential), "Expected base_mlp as nn.Sequential"
        self.d_model = d_model
        self.num_experts = num_experts

        # Build experts by copying base_mlp
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                QuickGELU(),
                nn.Linear(d_model * 4, d_model)
            )
            # upcycle init: copy weights from base
            expert[0].weight.data.copy_(base_mlp["c_fc"].weight.data)
            expert[0].bias.data.copy_(base_mlp["c_fc"].bias.data)
            # QuickGELU has no params
            expert[2].weight.data.copy_(base_mlp["c_proj"].weight.data)
            expert[2].bias.data.copy_(base_mlp["c_proj"].bias.data)
            self.experts.append(expert)

        # Router
        self.router = nn.Linear(d_model, num_experts)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        # Buffer for aux loss per forward
        self.aux_loss = None

        # Freeze experts by default; training policy handled externally
        for p in self.experts.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch, d_model]
        s, b, d = x.shape
        x_flat = x.reshape(s * b, d)

        # Router in fp32 for stability
        logits = self.router(x_flat.float())  # [T, E]
        probs = F.softmax(logits, dim=-1)     # [T, E]

        # Compute expert outputs (use input dtype)
        outs = []
        for expert in self.experts:
            outs.append(expert(x_flat))  # [T, d]
        outs = torch.stack(outs, dim=-1)  # [T, d, E]

        # Combine by probabilities
        probs = probs.to(outs.dtype)         # match dtype
        y = torch.einsum('tde,te->td', outs, probs)  # [T, d]
        y = y.view(s, b, d)

        # Aux load balancing loss: KL(mean_prob || uniform)
        mean_prob = probs.mean(dim=0)  # [E]
        uniform = torch.full_like(mean_prob, 1.0 / self.num_experts)
        kl = torch.sum(mean_prob * (torch.log(mean_prob + 1e-9) - torch.log(uniform)))
        self.aux_loss = kl

        return y


def apply_moe(args, clip_model: nn.Module):
    """Apply MoE to the last N layers of the text transformer in CLIP."""
    if not getattr(args, 'moe_enabled', False):
        return []

    # Access text transformer blocks
    if not hasattr(clip_model, 'transformer'):
        raise RuntimeError('CLIP model missing transformer for text tower')
    blocks = clip_model.transformer.resblocks
    num_layers = len(blocks)
    n_apply = min(getattr(args, 'moe_layers', 2), num_layers)
    start_idx = num_layers - n_apply

    replaced = []
    for li in range(start_idx, num_layers):
        block = blocks[li]
        base_mlp = block.mlp
        # Infer d_model from base_mlp
        d_model = base_mlp["c_fc"].in_features
        moe_mlp = MoEMLP(base_mlp, d_model, getattr(args, 'moe_num_experts', 4))
        block.mlp = moe_mlp
        replaced.append((li, moe_mlp))
    return replaced


def mark_moe_router_trainable(model: nn.Module, train_router_only: bool = True):
    """Set requires_grad for MoE components."""
    for m in model.modules():
        if isinstance(m, MoEMLP):
            # Router always trainable
            for p in m.router.parameters():
                p.requires_grad = True
            # Experts trainable only if not router-only
            for p in m.experts.parameters():
                p.requires_grad = (not train_router_only)


def get_router_parameters(model: nn.Module) -> List[nn.Parameter]:
    params = []
    for m in model.modules():
        if isinstance(m, MoEMLP):
            params += list(m.router.parameters())
    return params


def sum_moe_aux_loss(model: nn.Module) -> torch.Tensor:
    total = None
    for m in model.modules():
        if isinstance(m, MoEMLP) and (m.aux_loss is not None):
            total = m.aux_loss if total is None else (total + m.aux_loss)
    if total is None:
        total = torch.tensor(0.0, device=next(model.parameters()).device)
    return total