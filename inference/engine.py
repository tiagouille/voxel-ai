import os
import sys
import time
from typing import List, Dict, Optional, Generator, Tuple
import torch
import torch.nn.functional as F

# Support imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import VoxelConfig, ModelConfig
from model.architecture import VoxelTransformer
from tokenizer.voxel_tokenizer import VoxelTokenizer

class VoxelInferenceEngine:
    """Moteur d'inférence haute performance optimisé pour CPU ARM64 (Qualcomm Snapdragon X Plus) et GPU."""

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_dir: str = "tokenizer",
        device: str = "cpu",
        dtype: str = "float16",
        num_threads: int = 8
    ):
        self.device = torch.device(device)
        self.dtype_str = dtype

        # Optimisation CPU multithread (Snapdragon X Plus possède 8 cœurs haute performance)
        if device == "cpu":
            # Si float16 n'est pas supporté nativement sur certains CPU, bascule automatique sur float32 ou bfloat16
            if dtype == "float16" and not torch.cuda.is_available():
                # PyTorch CPU supporte bfloat16 et float32 efficacement
                self.dtype = torch.float32
            elif dtype == "bfloat16":
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float32

            torch.set_num_threads(num_threads)
            print(f"[Inférence ARM64/CPU] Configuration de {num_threads} threads CPU et type {self.dtype}.")
        else:
            self.dtype = torch.float16 if dtype == "float16" else torch.bfloat16

        # 1. Chargement du Tokenizer
        self.tokenizer = VoxelTokenizer.load(tokenizer_dir)

        # 2. Chargement des poids du modèle
        print(f"[*] Chargement du modèle depuis {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        config_dict = ckpt.get("config", {})
        # Instanciation de la configuration
        m_cfg = ModelConfig(**config_dict) if config_dict else ModelConfig(vocab_size=self.tokenizer.vocab_size)
        m_cfg.vocab_size = self.tokenizer.vocab_size

        self.model = VoxelTransformer(
            vocab_size=m_cfg.vocab_size,
            d_model=m_cfg.d_model,
            n_layers=m_cfg.n_layers,
            n_heads=m_cfg.n_heads,
            n_kv_heads=m_cfg.n_kv_heads,
            d_ff=m_cfg.d_ff,
            max_seq_len=m_cfg.max_seq_len,
            tie_word_embeddings=m_cfg.tie_word_embeddings,
            norm_eps=m_cfg.norm_eps,
            rope_theta=m_cfg.rope_theta,
            dropout=0.0
        )

        raw_sd = ckpt.get("model_state_dict", ckpt)
        state_dict = {}
        for k, v in raw_sd.items():
            nk = k
            nk = nk.replace("attention_norm", "attn_norm")
            nk = nk.replace("attention.wq", "attn.w_q")
            nk = nk.replace("attention.wk", "attn.w_k")
            nk = nk.replace("attention.wv", "attn.w_v")
            nk = nk.replace("attention.wo", "attn.w_o")
            nk = nk.replace("feed_forward.w1", "mlp.w_gate")
            nk = nk.replace("feed_forward.w2", "mlp.w_down")
            nk = nk.replace("feed_forward.w3", "mlp.w_up")
            if nk.startswith("output."):
                nk = nk.replace("output.", "lm_head.")
            state_dict[nk] = v
        self.model.load_state_dict(state_dict)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        print(f"[+] Modèle Voxel AI chargé en mémoire avec succès sur {self.device} ({self.dtype}).")

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15
    ) -> int:
        """Échantillonnage avec température, top-k, top-p et pénalité de répétition."""
        logits = logits.squeeze(0).float()  # (vocab_size,)

        # Pénalité de répétition
        if repetition_penalty != 1.0 and generated_tokens:
            for token_id in set(generated_tokens):
                if logits[token_id] < 0:
                    logits[token_id] *= repetition_penalty
                else:
                    logits[token_id] /= repetition_penalty

        # Si température très faible -> Greedy decoding (Argmax)
        if temperature < 1e-4:
            return int(torch.argmax(logits).item())

        logits = logits / temperature

        # Top-K filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-P (Nucleus) filtering
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Masquer les tokens au-delà du seuil cumulatif top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")

        # Distribution softmax et tirage multinomial
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())

    @torch.inference_mode()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        stop_tokens: Optional[List[int]] = None
    ) -> Generator[str, None, None]:
        """Génération incrémentale en streaming mot à mot avec accélération KV-Cache."""
        if stop_tokens is None:
            stop_tokens = [self.tokenizer.eos_token_id, self.tokenizer.endoftext_token_id]

        # Encodage du prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not input_ids:
            input_ids = [self.tokenizer.bos_token_id]

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated_tokens = []
        kv_caches = None

        # 1. Étape de pré-remplissage (Prefill) : traitement du prompt complet
        logits, _, kv_caches = self.model(input_tensor, kv_caches=None, start_pos=0)
        next_token_logits = logits[:, -1, :]
        next_token = self._sample_next_token(
            next_token_logits,
            generated_tokens,
            temperature, top_k, top_p, repetition_penalty
        )

        current_pos = len(input_ids)

        # 2. Étape de décodage autorégressif (Token par Token avec KV-Cache)
        for _ in range(max_new_tokens):
            if next_token in stop_tokens:
                break

            generated_tokens.append(next_token)
            token_str = self.tokenizer.decode([next_token], skip_special_tokens=False)

            # Ne pas émettre les balises spéciales brutes si elles apparaissent
            if "</s>" in token_str or "<|endoftext|>" in token_str:
                break

            yield token_str

            # Inférence pour un seul nouveau token grâce au KV-cache !
            next_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            logits, _, kv_caches = self.model(
                next_input,
                kv_caches=kv_caches,
                start_pos=current_pos
            )
            next_token_logits = logits[:, -1, :]
            current_pos += 1

            next_token = self._sample_next_token(
                next_token_logits,
                generated_tokens,
                temperature, top_k, top_p, repetition_penalty
            )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15
    ) -> str:
        """Génération complète non-streaming."""
        response = ""
        for chunk in self.stream_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        ):
            response += chunk
        return response.strip()
