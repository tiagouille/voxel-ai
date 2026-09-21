import os
import json
from typing import List, Dict, Union, Optional
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

SPECIAL_TOKENS = [
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endoftext|>"
]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SYSTEM_ID = 4
USER_ID = 5
ASSISTANT_ID = 6
ENDOFTEXT_ID = 7

class VoxelTokenizer:
    """Tokenizer BPE Byte-Level personnalisé pour Voxel AI."""

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        self._tokenizer = tokenizer
        self.pad_token_id = PAD_ID
        self.bos_token_id = BOS_ID
        self.eos_token_id = EOS_ID
        self.unk_token_id = UNK_ID
        self.system_token_id = SYSTEM_ID
        self.user_token_id = USER_ID
        self.assistant_token_id = ASSISTANT_ID
        self.endoftext_token_id = ENDOFTEXT_ID

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is None:
            return 0
        return self._tokenizer.get_vocab_size()

    @classmethod
    def train_from_files(
        cls,
        files: List[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ) -> "VoxelTokenizer":
        """Entraîne un tokenizer BPE Byte-Level à partir d'une liste de fichiers texte."""
        if special_tokens is None:
            special_tokens = SPECIAL_TOKENS

        # Initialisation du modèle BPE Byte-Level
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        tokenizer.train(files=files, trainer=trainer)
        return cls(tokenizer=tokenizer)

    def save(self, save_directory: str):
        """Sauvegarde le tokenizer au format tokenizer.json et vocab.json."""
        os.makedirs(save_directory, exist_ok=True)
        tokenizer_path = os.path.join(save_directory, "tokenizer.json")
        self._tokenizer.save(tokenizer_path)

        # Sauvegarder également les métadonnées et identifiants spéciaux
        meta = {
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
            "system_token_id": self.system_token_id,
            "user_token_id": self.user_token_id,
            "assistant_token_id": self.assistant_token_id
        }
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, save_directory_or_file: str) -> "VoxelTokenizer":
        """Charge un tokenizer sauvegardé."""
        if os.path.isdir(save_directory_or_file):
            file_path = os.path.join(save_directory_or_file, "tokenizer.json")
        else:
            file_path = save_directory_or_file

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier tokenizer introuvable: {file_path}")

        tok = Tokenizer.from_file(file_path)
        return cls(tokenizer=tok)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Convertit une chaîne de caractères en une liste d'identifiants de tokens."""
        if self._tokenizer is None:
            raise ValueError("Tokenizer non initialisé.")
        
        enc = self._tokenizer.encode(text)
        ids = enc.ids
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Décode une liste de tokens en texte lisible."""
        if self._tokenizer is None:
            raise ValueError("Tokenizer non initialisé.")
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True
    ) -> str:
        """Formate une liste de messages conversationnels en prompt textuel structuré."""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if role == "system":
                formatted += f"<|system|>\n{content}</s>\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}</s>\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}</s>\n"

        if add_generation_prompt:
            formatted += "<|assistant|>\n"

        return formatted
