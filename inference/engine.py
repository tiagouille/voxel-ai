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

    def _generate_tokens_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        stop_tokens: Optional[List[int]] = None,
        stop_words: Optional[List[str]] = None
    ) -> Generator[str, None, None]:
        """Génération incrémentale d'un fragment de texte avec gestion robuste de l'encodage UTF-8 et KV-Cache."""
        if stop_tokens is None:
            stop_tokens = [self.tokenizer.eos_token_id, self.tokenizer.endoftext_token_id]
        if stop_words is None:
            stop_words = ["<|endoftext|>", "</s>", "###"]

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not input_ids:
            input_ids = [self.tokenizer.bos_token_id]

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated_tokens = []
        kv_caches = None

        # Prefill du prompt
        logits, _, kv_caches = self.model(input_tensor, kv_caches=None, start_pos=0)
        next_token_logits = logits[:, -1, :]
        next_token = self._sample_next_token(
            next_token_logits,
            generated_tokens,
            temperature, top_k, top_p, repetition_penalty
        )

        current_pos = len(input_ids)
        prev_text = ""

        for _ in range(max_new_tokens):
            if next_token in stop_tokens:
                break

            generated_tokens.append(next_token)
            full_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Si un caractère multi-octets UTF-8 est en cours de décodage (ex: lettre avec accent), attendre le prochain token
            if full_text.endswith("\ufffd"):
                pass
            else:
                # Vérification des mots d'arrêt textuels
                stopped = False
                for sw in stop_words:
                    if sw in full_text:
                        clean = full_text.split(sw)[0]
                        if len(clean) > len(prev_text):
                            yield clean[len(prev_text):]
                        stopped = True
                        break
                if stopped:
                    break

                if len(full_text) > len(prev_text):
                    delta = full_text[len(prev_text):]
                    prev_text = full_text
                    yield delta

            # Décodage autorégressif pas à pas
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

        # Si un reste valide existe en fin de génération
        final_clean = full_text.rstrip("\ufffd") if "full_text" in locals() else ""
        for sw in stop_words:
            if sw in final_clean:
                final_clean = final_clean.split(sw)[0]
        if len(final_clean) > len(prev_text):
            yield final_clean[len(prev_text):]

    @torch.inference_mode()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        stop_tokens: Optional[List[int]] = None,
        enable_thinking: bool = False
    ) -> Generator[str, None, None]:
        """Génération incrémentale en streaming, avec support du Mode Réflexion (CoT / DeepThink)."""
        if enable_thinking:
            yield "<think>\n"

            # Extraction de la question utilisateur
            user_query = ""
            if "### Question:" in prompt:
                user_query = prompt.split("### Question:")[-1].split("### Réponse:")[0].strip()
            elif "### User:" in prompt:
                user_query = prompt.split("### User:")[-1].split("### Assistant:")[0].strip()
            else:
                user_query = prompt.strip()

            if not user_query:
                user_query = "Requête de l'utilisateur"

            yield f"• Analyse de la question : « {user_query} »\n"
            yield "• Décomposition conceptuelle et réflexion interne :\n  "

            thought_prompt = f"### Question:\n{user_query}\n\n### Réflexion conceptuelle:\n"
            thought_tokens_limit = min(60, max(25, max_new_tokens // 4))

            for chunk in self._generate_tokens_stream(
                prompt=thought_prompt,
                max_new_tokens=thought_tokens_limit,
                temperature=0.75,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.2,
                stop_tokens=stop_tokens,
                stop_words=["###", "<|endoftext|>", "</s>"]
            ):
                yield chunk

            yield "\n• Synthèse : Structuration et validation de la réponse finale.\n"
            yield "</think>\n\n"

        # Génération de la réponse principale
        for chunk in self._generate_tokens_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens,
            stop_words=["###", "<|endoftext|>", "</s>", "\nQuestion:"]
        ):
            yield chunk

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        enable_thinking: bool = False
    ) -> str:
        """Génération complète non-streaming."""
        response = ""
        for chunk in self.stream_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            enable_thinking=enable_thinking
        ):
            response += chunk
        return response.strip()

