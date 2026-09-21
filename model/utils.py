import sys
import os
from typing import Dict, Any
import torch
import torch.nn as nn

# Support d'importation relative et absolue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import VoxelConfig, ModelConfig
from model.architecture import VoxelTransformer

def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """Compte avec précision les paramètres du modèle et affiche un bilan détaillé."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Détail par sous-modules
    breakdown = {}
    for name, module in model.named_children():
        sub_params = sum(p.numel() for p in module.parameters())
        breakdown[name] = sub_params

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "total_millions": total_params / 1e6,
        "breakdown": breakdown
    }

def print_model_summary(model: nn.Module, config: ModelConfig):
    """Affiche un résumé soigné de l'architecture et du comptage de paramètres."""
    counts = count_parameters(model)
    print("=" * 65)
    print(f"       ARCHITECTURE VOXEL AI - SPÉCIFICATIONS MODÈLE")
    print("=" * 65)
    print(f" Modèle                  : {config.name}")
    print(f" Taille du vocabulaire   : {config.vocab_size:,}")
    print(f" Dimension cachée (d)    : {config.d_model}")
    print(f" Nombre de couches       : {config.n_layers}")
    print(f" Têtes Query (Q)         : {config.n_heads} (dim={config.head_dim})")
    print(f" Têtes Key/Value (KV)    : {config.n_kv_heads} (GQA ratio={config.n_heads//config.n_kv_heads}:1)")
    print(f" Dimension SwiGLU (d_ff) : {config.d_ff}")
    print(f" Contexte maximum        : {config.max_seq_len} tokens")
    print(f" Poids partagés (tied)   : {config.tie_word_embeddings}")
    print("-" * 65)
    print(" DÉCOMPTE DÉTAILLÉ DES PARAMÈTRES :")
    for name, p_count in counts["breakdown"].items():
        pct = (p_count / counts["total_parameters"]) * 100
        print(f"  • {name:20s}: {p_count:>12,} ({p_count/1e6:>6.2f} M) [{pct:>5.1f}%]")
    print("-" * 65)
    print(f" TOTAL PARAMÈTRES         : {counts['total_parameters']:,} ({counts['total_millions']:.2f} Millions)")
    print(f" TOTAL ENTRAÎNABLES       : {counts['trainable_parameters']:,}")
    print("=" * 65)

def build_model_from_config(config: VoxelConfig) -> VoxelTransformer:
    """Instancie le modèle VoxelTransformer à partir de la configuration."""
    m = config.model
    model = VoxelTransformer(
        vocab_size=m.vocab_size,
        d_model=m.d_model,
        n_layers=m.n_layers,
        n_heads=m.n_heads,
        n_kv_heads=m.n_kv_heads,
        d_ff=m.d_ff,
        max_seq_len=m.max_seq_len,
        tie_word_embeddings=m.tie_word_embeddings,
        norm_eps=m.norm_eps,
        rope_theta=m.rope_theta,
        dropout=m.dropout,
        initializer_range=m.initializer_range
    )
    return model

if __name__ == "__main__":
    # Test autonome et vérification directe du nombre de paramètres
    cfg = ModelConfig()
    print("Vérification analytique :")
    analytic = cfg.calculate_parameter_count()
    print(f"Paramètres calculés analytiquement : {analytic['total_parameters']:,} ({analytic['total_parameters']/1e6:.2f} M)")
    
    print("\nInstanciation PyTorch du modèle...")
    model = VoxelTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        tie_word_embeddings=cfg.tie_word_embeddings
    )
    print_model_summary(model, cfg)
