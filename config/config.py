import json
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
    name: str = "Voxel-AI-500M"
    vocab_size: int = 32000
    d_model: int = 1536
    n_layers: int = 16
    n_heads: int = 24
    n_kv_heads: int = 8
    d_ff: int = 4096
    max_seq_len: int = 2048
    tie_word_embeddings: bool = False
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0
    initializer_range: float = 0.02

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        return self.d_model // self.n_heads

    def calculate_parameter_count(self) -> Dict[str, int]:
        """Calcul analytique précis du nombre de paramètres de l'architecture."""
        head_dim = self.head_dim
        # 1. Embeddings
        token_embed = self.vocab_size * self.d_model
        lm_head = 0 if self.tie_word_embeddings else (self.vocab_size * self.d_model)

        # 2. Attention par couche (GQA)
        q_proj = self.d_model * (self.n_heads * head_dim)
        k_proj = self.d_model * (self.n_kv_heads * head_dim)
        v_proj = self.d_model * (self.n_kv_heads * head_dim)
        o_proj = (self.n_heads * head_dim) * self.d_model
        attn_per_layer = q_proj + k_proj + v_proj + o_proj

        # 3. Normalisation RMSNorm (2 par couche)
        norm_per_layer = 2 * self.d_model

        # 4. Feed-Forward SwiGLU (gate, up, down)
        gate_proj = self.d_model * self.d_ff
        up_proj = self.d_model * self.d_ff
        down_proj = self.d_ff * self.d_model
        mlp_per_layer = gate_proj + up_proj + down_proj

        layer_total = attn_per_layer + norm_per_layer + mlp_per_layer
        all_layers = self.n_layers * layer_total

        # 5. RMSNorm finale
        final_norm = self.d_model

        total = token_embed + all_layers + final_norm + lm_head
        return {
            "token_embed": token_embed,
            "lm_head": lm_head,
            "attn_per_layer": attn_per_layer,
            "mlp_per_layer": mlp_per_layer,
            "layer_total": layer_total,
            "all_layers": all_layers,
            "final_norm": final_norm,
            "total_parameters": total
        }

@dataclass
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 50000
    eval_interval: int = 500
    eval_steps: int = 50
    save_interval: int = 1000
    mixed_precision: str = "auto"
    gradient_checkpointing: bool = True
    seed: int = 42

@dataclass
class InferenceConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.15
    max_new_tokens: int = 512
    dtype: str = "float16"
    device: str = "cpu"

@dataclass
class VoxelConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    checkpoint_dir: str = "checkpoints"
    tokenizer_dir: str = "tokenizer"
    dataset_dir: str = "dataset/data"
    drive_checkpoint_dir: str = "/content/drive/MyDrive/voxel-ai/checkpoints"

    @classmethod
    def load_from_json(cls, config_path: str) -> "VoxelConfig":
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        m_cfg = ModelConfig(**data.get("model", {}))
        t_cfg = TrainingConfig(**data.get("training", {}))
        i_cfg = InferenceConfig(**data.get("inference", {}))
        paths = data.get("paths", {})
        
        return cls(
            model=m_cfg,
            training=t_cfg,
            inference=i_cfg,
            checkpoint_dir=paths.get("checkpoint_dir", "checkpoints"),
            tokenizer_dir=paths.get("tokenizer_dir", "tokenizer"),
            dataset_dir=paths.get("dataset_dir", "dataset/data"),
            drive_checkpoint_dir=paths.get("drive_checkpoint_dir", "/content/drive/MyDrive/voxel-ai/checkpoints")
        )

    def save_to_json(self, config_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        data = {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "inference": self.inference.__dict__,
            "paths": {
                "checkpoint_dir": self.checkpoint_dir,
                "tokenizer_dir": self.tokenizer_dir,
                "dataset_dir": self.dataset_dir,
                "drive_checkpoint_dir": self.drive_checkpoint_dir
            }
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
