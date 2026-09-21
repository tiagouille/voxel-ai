import math
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Pré-calcul des fréquences pour le Rotary Position Embedding (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # nombres complexes e^(i*theta)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applique les rotations RoPE sur les tenseurs de Query et de Key."""
    # xq shape: (bsz, seqlen, n_heads, head_dim)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Adapter la forme de freqs_cis pour le broadcasting
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seqlen, 1, head_dim//2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLUMLP(nn.Module):
    """Feed-Forward Network avec activation SwiGLU (Shazeer, 2020)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) avec support KV-Cache pour l'inférence ultra-rapide."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.num_queries_per_kv = n_heads // n_kv_heads
        self.dropout = dropout

        self.w_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seqlen, _ = x.shape

        # Projections linéaires
        xq = self.w_q(x)
        xk = self.w_k(x)
        xv = self.w_v(x)

        # Reshape en têtes : (bsz, seqlen, n_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Application de RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Gestion du KV-Cache pour l'inférence incrémentale
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            # Concaténation le long de la dimension de séquence
            xk = torch.cat([prev_k, xk], dim=1)
            xv = torch.cat([prev_v, xv], dim=1)
            new_kv_cache = (xk, xv)
        else:
            new_kv_cache = (xk, xv) if not self.training else None

        # Format requis pour F.scaled_dot_product_attention : (bsz, n_heads, total_seqlen, head_dim)
        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bsz, n_kv_heads, total_seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bsz, n_kv_heads, total_seqlen, head_dim)

        # Répétition des clés/valeurs si n_kv_heads < n_heads (GQA)
        if self.num_queries_per_kv > 1:
            xk = xk.repeat_interleave(self.num_queries_per_kv, dim=1)
            xv = xv.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Attention Scaled Dot Product (FlashAttention / SDPA natif)
        is_causal = (mask is None and seqlen > 1 and kv_cache is None)
        dropout_p = self.dropout if self.training else 0.0

        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )

        # Remise en forme et projection de sortie
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.w_o(output), new_kv_cache


class TransformerBlock(nn.Module):
    """Bloc Transformer individuel combinant RMSNorm, GQA et SwiGLU MLP."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        norm_eps: float = 1e-5,
        dropout: float = 0.0
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout=dropout)
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.mlp = SwiGLUMLP(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-Norm Attention
        norm_x = self.attn_norm(x)
        h, new_kv = self.attn(norm_x, freqs_cis, mask=mask, kv_cache=kv_cache, start_pos=start_pos)
        x = x + h

        # Pre-Norm MLP
        x = x + self.mlp(self.ffn_norm(x))
        return x, new_kv


class VoxelTransformer(nn.Module):
    """Architecture complète Voxel AI (Transformer Decoder ~500M)."""
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 1536,
        n_layers: int = 16,
        n_heads: int = 24,
        n_kv_heads: int = 8,
        d_ff: int = 4096,
        max_seq_len: int = 2048,
        tie_word_embeddings: bool = False,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        initializer_range: float = 0.02
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = False

        # Embeddings de jetons
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)

        # Couches Transformer
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                d_ff=d_ff,
                norm_eps=norm_eps,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Normalisation finale
        self.norm = RMSNorm(d_model, eps=norm_eps)

        # Tête de projection du vocabulaire
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        # Pré-calcul des fréquences RoPE (sur CPU au départ)
        head_dim = d_model // n_heads
        self.freqs_cis = precompute_freqs_cis(head_dim, max_seq_len, theta=rope_theta)

        # Initialisation soignée des poids
        self.apply(lambda module: self._init_weights(module, initializer_range))

    def _init_weights(self, module: nn.Module, std: float = 0.02):
        """Initialisation aléatoire des poids (Gaussienne tronquée / normale)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def enable_gradient_checkpointing(self, enable: bool = True):
        self.gradient_checkpointing = enable

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass du modèle.
        Args:
            input_ids: Tenseur (bsz, seqlen) d'identifiants de tokens
            targets: Tenseur optionnel (bsz, seqlen) pour calculer la perte
            kv_caches: Liste optionnelle de tuples (K, V) pour chaque couche
            start_pos: Index de départ de la séquence pour le RoPE lors de la génération incrémentale
        """
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        # Extraire la portion de freqs_cis correspondant à la position
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen].to(device)

        # Embeddings d'entrée
        h = self.tok_embeddings(input_ids)

        new_kv_caches = [] if kv_caches is not None or not self.training else None

        # Masque causal pour la génération incrémentale si seqlen > 1
        mask = None
        if seqlen > 1 and kv_caches is not None:
            # En cours de pré-remplissage avec cache existant
            total_len = start_pos + seqlen
            mask = torch.full((seqlen, total_len), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=start_pos + 1)

        # Traversée des couches
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            
            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing pour économiser la mémoire VRAM
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                h, new_cache = checkpoint(
                    create_custom_forward(layer),
                    h, freqs_cis, mask, layer_cache, start_pos,
                    use_reentrant=False
                )
            else:
                h, new_cache = layer(h, freqs_cis, mask=mask, kv_cache=layer_cache, start_pos=start_pos)

            if new_kv_caches is not None:
                new_kv_caches.append(new_cache)

        # RMSNorm finale
        h = self.norm(h)

        # Calcul des logits
        logits = self.lm_head(h)

        # Calcul de la perte causale si targets est fourni
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss, new_kv_caches
