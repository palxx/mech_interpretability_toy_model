%%writefile toy_mech_interp.py

"""
Toy Mechanistic Interpretability Transformer (PyTorch)
-----------------------------------------------------------------
A tiny, fully self-contained Transformer you can *mechanistically* analyze.
It trains on a synthetic next-letter task where the correct next token is
(prev_token + 1) mod V over a small alphabet. A single attention head can
solve this by attending to the previous token (a classic "bigram/induction"
style circuit). The code includes:

- Minimal tokenizer & synthetic dataset
- Tiny Transformer (1–2 layers, configurable heads)
- Training loop (AdamW)
- Activation cache during forward passes
- Logit lens & direct logit attribution to components of the residual stream
- Attention and OV/QK circuit inspection utilities
- Activation patching (clean→corrupt) for causal analysis

Usage
-----
python toy_mech_interp.py --layers 1 --heads 1 --d_model 64 --seq_len 32 \
  --batch_size 128 --steps 3000 --device cuda

After training, it runs a demo with:
- Attention pattern visualization (printed as small matrices)
- Logit lens & per-component logit attribution
- OV-circuit probe: does the head map token i to i+1?
- Activation patching: shows which head/position carries the causal signal

You can then modify TASKS to try:
- skip-1 or skip-2 bigram (next token = token_{i-k})
- periodic addition (e.g., +3 mod V)
- simple copy tasks
"""
from __future__ import annotations
import math
# import argparse # Removed argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Tokenizer / Data
# ------------------------------

class ToyTokenizer:
    def __init__(self, alphabet: str = "abcdefghijklmnopqrstuvwxyz"):
        self.alphabet = alphabet
        self.BOS = "<BOS>"
        self.vocab = [self.BOS] + list(alphabet)
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

# Synthetic task: next token is (prev + k) mod V, skipping the BOS token.
@dataclass
class TaskConfig:
    k: int = 1               # how many steps ahead mod alphabet size
    seq_len: int = 32
    batch_size: int = 128

class NextLetterDataset:
    def __init__(self, tok: ToyTokenizer, cfg: TaskConfig):
        self.tok = tok
        self.cfg = cfg
        self.V = tok.vocab_size
        assert self.V >= 3, "Need at least BOS + 2 letters"

    def sample_batch(self, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = self.cfg.batch_size, self.cfg.seq_len
        # sequences: start with BOS, then letters chosen uniformly at random
        # The correct target at position i is (token_{i-1}+k) mod V (excluding BOS wrap)
        # We'll map only letters (1..V-1) via mod V-1, keeping BOS only at pos 0.
        x = torch.zeros((B, L), dtype=torch.long)
        x[:, 0] = 0  # BOS
        # pick a random start letter per sequence
        starts = torch.randint(1, self.V, (B,))
        for b in range(B):
            for i in range(1, L):
                # deterministically increase from the start
                x[b, i] = 1 + ((starts[b] - 1 + (i-0)) % (self.V - 1))
        # targets are next = prev + k
        y = torch.zeros_like(x)
        y[:, 0] = x[:, 0]  # arbitrary; we don't train on pos 0
        for b in range(B):
            for i in range(1, L):
                prev = x[b, i-1]
                if prev == 0:
                    y[b, i] = 0
                else:
                    y[b, i] = 1 + (((prev - 1) + self.cfg.k) % (self.V - 1))
        return x.to(device), y.to(device)

# ------------------------------
# Model
# ------------------------------

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        xhat = (x - mu) / torch.sqrt(var + self.eps)
        return xhat * self.weight + self.bias

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.W_Q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor, cache: Optional[Dict]=None, layer_idx: int=0) -> torch.Tensor:
        B, L, D = x.shape
        H, d = self.n_heads, self.d_head
        q = self.W_Q(x).view(B, L, H, d).transpose(1, 2)  # B,H,L,d
        k = self.W_K(x).view(B, L, H, d).transpose(1, 2)  # B,H,L,d
        v = self.W_V(x).view(B, L, H, d).transpose(1, 2)  # B,H,L,d
        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)  # B,H,L,L
        att = att_scores.softmax(dim=-1)
        z = att @ v  # B,H,L,d
        z = z.transpose(1, 2).contiguous().view(B, L, H * d)
        out = self.W_O(z)  # B,L,D

        if cache is not None:
            cache[f"layer{layer_idx}.q"] = q.detach()
            cache[f"layer{layer_idx}.k"] = k.detach()
            cache[f"layer{layer_idx}.v"] = v.detach()
            cache[f"layer{layer_idx}.att"] = att.detach()
            cache[f"layer{layer_idx}.z"] = z.detach()
            cache[f"layer{layer_idx}.att_out"] = out.detach()
        return out

class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model)
    def forward(self, x: torch.Tensor, cache: Optional[Dict]=None, layer_idx: int=0) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        out = self.fc2(h)
        if cache is not None:
            cache[f"layer{layer_idx}.mlp_h"] = h.detach()
            cache[f"layer{layer_idx}.mlp_out"] = out.detach()
        return out

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_mlp: int):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, d_head)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)
    def forward(self, x: torch.Tensor, cache: Optional[Dict], layer_idx: int) -> torch.Tensor:
        resid_pre = self.ln1(x)
        att = self.attn(resid_pre, cache=cache, layer_idx=layer_idx)
        x = x + att
        if cache is not None:
            cache[f"layer{layer_idx}.resid_after_attn"] = x.detach()
        resid_mid = self.ln2(x)
        mlp_out = self.mlp(resid_mid, cache=cache, layer_idx=layer_idx)
        x = x + mlp_out
        if cache is not None:
            cache[f"layer {layer_idx}.resid_post"] = x.detach() # Fixed typo here
        return x

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int=64, n_layers: int=1, n_heads: int=1, d_head: int=64, d_mlp: int=128, max_seq_len: int=128, tie_embeddings: bool=True):
        super().__init__()
        assert d_model == n_heads * d_head, "d_model must equal n_heads*d_head"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_head, d_mlp) for _ in range(n_layers)
        ])
        self.ln_final = LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.unembed.weight = self.embed.weight  # weight tying

    def forward(self, tokens: torch.Tensor, cache: Optional[Dict]=None) -> torch.Tensor:
        B, L = tokens.shape
        pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.embed(tokens) + self.pos(pos_ids)
        if cache is not None:
            cache["resid_pre.0"] = x.detach()
        for i, block in enumerate(self.blocks):
            x = block(x, cache=cache, layer_idx=i)
        x = self.ln_final(x)
        if cache is not None:
            cache["resid_final"] = x.detach()
        logits = self.unembed(x)
        if cache is not None:
            cache["logits"] = logits.detach()
        return logits

# ------------------------------
# Training
# ------------------------------

def train(model: TinyTransformer, data: NextLetterDataset, steps: int=2000, lr: float=1e-3, device: str="cpu", print_every: int=200) -> None:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_ema = None
    for step in range(1, steps+1):
        x, y = data.sample_batch(device)
        logits = model(x)
        # ignore loss at position 0 (BOS)
        loss = F.cross_entropy(logits[:, 1:].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_ema = loss.item() if loss_ema is None else 0.98*loss_ema + 0.02*loss.item()
        if (step % print_every) == 0:
            print(f"step {step:5d} | loss {loss_ema:.4f}")

# ------------------------------
# Interpretability Utilities
# ------------------------------

@torch.no_grad()
def run_with_cache(model: TinyTransformer, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    cache: Dict[str, torch.Tensor] = {}
    logits = model(tokens, cache=cache)
    return logits, cache

@torch.no_grad()
def attention_patterns(cache: Dict[str, torch.Tensor], layer: int=0) -> torch.Tensor:
    return cache[f"layer{layer}.att"]  # B,H,L,L

@torch.no_grad()
def logit_lens(model: TinyTransformer, cache: Dict[str, torch.Tensor], positions: List[int]) -> Dict[str, torch.Tensor]:
    """ Project intermediate residuals through unembed to get pseudo-logits. """
    out = {}
    for i in range(model.n_layers):
        resid = cache.get(f"layer{i}.resid_after_attn", None)
        if resid is not None:
            logits_i = model.unembed(model.ln_final(resid))
            out[f"after_attn_L{i}"] = logits_i[:, positions]
        resid_post = cache.get(f"layer{i}.resid_post", None)
        if resid_post is not None:
            logits_i = model.unembed(model.ln_final(resid_post))
            out[f"post_mlp_L{i}"] = logits_i[:, positions]
    resid0 = cache.get("resid_pre.0")
    if resid0 is not None:
        out["embed+pos"] = model.unembed(model.ln_final(resid0))[:, positions]
    return out  # dict of [B,len(positions),V]

@torch.no_grad()
def direct_logit_attribution(model: TinyTransformer, cache: Dict[str, torch.Tensor], tokens: torch.Tensor, pos: int, target_id: int) -> Dict[str, float]:
    """Decompose final logit for `target_id` at position `pos` into components.
    We project each component vector through final layernorm & unembed.
    Components: embed+pos, each layer's attn output, each layer's mlp output.
    """
    LN = model.ln_final
    U = model.unembed.weight  # [V, D]
    contribs: Dict[str, float] = {}

    def proj(vec: torch.Tensor) -> torch.Tensor:
        # vec: [D]; apply LN (with batch-like shape) then dot with U[target]
        v = vec.unsqueeze(0).unsqueeze(0)  # [1,1,D]
        v = LN(v)[0,0]
        return (U[target_id] @ v).item()

    # embed + pos at this position
    resid0 = cache["resid_pre.0"][0, pos]  # [D]
    contribs["embed+pos"] = proj(resid0)

    x = resid0.clone()
    for i in range(model.n_layers):
        att_out = cache[f"layer{i}.att_out"][0, pos]
        contribs[f"att_L{i}"] = proj(att_out)
        x = x + att_out
        mlp_out = cache[f"layer{i}.mlp_out"][0, pos]
        contribs[f"mlp_L{i}"] = proj(mlp_out)
        x = x + mlp_out

    # Final residual contribution equals sum of components; check consistency
    resid_final = cache["resid_final"][0, pos]
    total = (U[target_id] @ model.ln_final(resid_final)).item()
    contribs["TOTAL (check)"] = total
    return contribs

# ---------- OV / QK circuit probes ----------
@torch.no_grad()
def ov_probe(model: TinyTransformer, head_layer: int=0, n_samples: int=5) -> None:
    """Check whether OV maps token i -> i+1 via U.
    We approximate by applying W_V then W_O then U to each token embedding.
    """
    block = model.blocks[head_layer]
    W_E = model.embed.weight.data  # [V,D]
    W_V = block.attn.W_V.weight.data  # [D, H*d]
    W_O = block.attn.W_O.weight.data  # [H*d, D]
    U = model.unembed.weight.data     # [V, D]
    D = model.d_model

    # For simplicity assume H*d == D and one head learns the mapping.
    M = (W_O @ W_V.T)  # [D,D] approximates OV path mapping from input to residual
    logits = (U @ M @ W_E.T)  # [V,V] map from input token id -> logits over tokens
    pred = logits.argmax(dim=-1)  # [V]

    print("OV probe: token -> argmax(U * W_O * W_V * E(token))")
    V = model.vocab_size
    for i in range(1, min(V, n_samples+1)):
        print(f" token {i} -> {int(pred[i])}")

# ---------- Activation patching ----------
@dataclass
class PatchSpec:
    layer: int
    kind: str   # 'z' (per-head pre-O), 'att_out' (post-O), 'mlp_out'
    positions: Optional[List[int]] = None  # positions to patch (sequence indices)

@torch.no_grad()
def forward_with_patching(model: TinyTransformer, tokens: torch.Tensor, source_cache: Dict[str, torch.Tensor], spec: PatchSpec) -> torch.Tensor:
    """Run forward, but patch a component from source_cache into the computation.
    For simplicity, we patch *after* it's computed at the module level.
    """
    cache: Dict[str, torch.Tensor] = {}

    def hook_attn(module: SelfAttention, input, output):
        if spec.kind == 'att_out' and module is model.blocks[spec.layer].attn:
            out = output.clone()
            src = source_cache[f"layer{spec.layer}.att_out"]  # [B,L,D]
            if spec.positions is None:
                out[:] = src
            else:
                out[:, spec.positions] = src[:, spec.positions]
            return out
        return None

    handles = []
    handles.append(model.blocks[spec.layer].attn.register_forward_hook(hook_attn))

    try:
        logits = model(tokens, cache=cache)
    finally:
        for h in handles:
            h.remove()
    return logits


# ------------------------------
# Demo / CLI (Modified for Colab)
# ------------------------------

# Hardcoded arguments for Colab execution
# p = argparse.ArgumentParser()
# p.add_argument('--layers', type=int, default=1)
# p.add_argument('--heads', type=int, default=1)
# p.add_argument('--d_model', type=int, default=64)
# p.add_argument('--d_head', type=int, default=64)
# p.add_argument('--d_mlp', type=int, default=128)
# p.add_argument('--seq_len', type=int, default=32)
# p.add_argument('--batch_size', type=int, default=128)
# p.add_argument('--steps', type=int, default=2000)
# p.add_argument('--lr', type=float, default=1e-3)
# p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
# args = p.parse_args()

# Hardcoded values
args_layers = 1
args_heads = 1
args_d_model = 64
args_d_head = 64
args_d_mlp = 128
args_seq_len = 32
args_batch_size = 128
args_steps = 2000
args_lr = 1e-3
args_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def demo(model: TinyTransformer, tok: ToyTokenizer, device: str):
    model.eval()
    # craft one example sequence
    seq_len = 20
    x = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    x[0,0] = 0
    start = random.randint(1, tok.vocab_size-1)
    for i in range(1, seq_len):
        x[0,i] = 1 + ((start - 1 + (i-0)) % (tok.vocab_size - 1))
    logits, cache = run_with_cache(model, x)
    pred = logits.argmax(dim=-1)

    # Show attention on a few last positions
    att = attention_patterns(cache, layer=0)[0]  # H,L,L
    print("\nAttention pattern (layer 0, head 0) [last 8x8 block]:")
    H, L, _ = att.shape
    for h in range(min(H, 1)):
        a = att[h, -8:, -8:].cpu().numpy()
        np.set_printoptions(precision=2, suppress=True)
        print(a)

    # Logit lens @ final few positions
    lens = logit_lens(model, cache, positions=list(range(seq_len-4, seq_len)))
    for name, ll in lens.items():
        # ll: [B,P,V]
        probs = ll.softmax(dim=-1)
        top = probs[0].topk(3, dim=-1).indices  # [P,3]
        print(f"\nLogit lens at {name}: top-3 token ids for last positions")
        print(top.cpu().numpy())

    # Direct logit attribution at the final position
    pos = seq_len - 1
    target = pred[0, pos].item()
    dla = direct_logit_attribution(model, cache, x, pos=pos, target_id=target)
    print("\nDirect Logit Attribution at last position (contrib to predicted token):")
    for k, v in dla.items():
        print(f"  {k:20s} : {v:+.3f}")

    # OV probe
    ov_probe(model, head_layer=0)

    # Activation patching: corrupt by rotating the sequence start
    x_corrupt = x.clone()
    # rotate the earlier part so prev-token signal is wrong for later positions
    x_corrupt[0, 1:seq_len//2] = (x_corrupt[0, 1:seq_len//2] - 1 - 3) % (tok.vocab_size)  # cheap corruption
    logits_clean, cache_clean = run_with_cache(model, x)
    logits_corrupt, cache_corrupt = run_with_cache(model, x_corrupt)

    # Patch att_out from clean into corrupt at all positions
    spec = PatchSpec(layer=0, kind='att_out', positions=None)
    logits_patched = forward_with_patching(model, x_corrupt, cache_clean, spec)

    def token_log_prob(logits, t, pos):
        return F.log_softmax(logits, dim=-1)[0, pos, t].item()

    target_last = x[0, -1].item()
    print("\nActivation Patching (att_out, L0): log-prob of correct last token")
    print(f"  clean   : {token_log_prob(logits_clean, target_last, seq_len-1):+.3f}")
    print(f"  corrupt : {token_log_prob(logits_corrupt, target_last, seq_len-1):+.3f}")
    print(f"  patched : {token_log_prob(logits_patched, target_last, seq_len-1):+.3f}")


def main():
    # Using hardcoded values instead of argparse
    # p = argparse.ArgumentParser()
    # p.add_argument('--layers', type=int, default=1)
    # p.add_argument('--heads', type=int, default=1)
    # p.add_argument('--d_model', type=int, default=64)
    # p.add_argument('--d_head', type=int, default=64)
    # p.add_argument('--d_mlp', type=int, default=128)
    # p.add_argument('--seq_len', type=int, default=32)
    # p.add_argument('--batch_size', type=int, default=128)
    # p.add_argument('--steps', type=int, default=2000)
    # p.add_argument('--lr', type=float, default=1e-3)
    # p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    # args = p.parse_args()

    # assert args.d_model == args.heads * args.d_head, "d_model must equal heads*d_head"
    assert args_d_model == args_heads * args_d_head, "d_model must equal heads*d_head"

    device = args_device
    tok = ToyTokenizer()
    dataset = NextLetterDataset(tok, TaskConfig(k=1, seq_len=args_seq_len, batch_size=args_batch_size))

    model = TinyTransformer(
        vocab_size=tok.vocab_size,
        d_model=args_d_model,
        n_layers=args_layers,
        n_heads=args_heads,
        d_head=args_d_head,
        d_mlp=args_d_mlp,
        max_seq_len=args_seq_len,
        tie_embeddings=True,
    ).to(device)

    print("Training...")
    train(model, dataset, steps=args_steps, lr=args_lr, device=device, print_every=max(50, args_steps//20))

    print("\nRunning demo analyses...")
    demo(model, tok, device)

if __name__ == '__main__':
    main()