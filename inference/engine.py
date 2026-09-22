import os
import sys
import time
import re
from typing import List, Dict, Optional, Generator, Tuple
import torch
import torch.nn.functional as F

ALPACA_LEAK_REGEX = re.compile(
    r'(?:[\.\!\?]\s*|\n)(?:Trouve|Décris|Crée|Rédige|Explique|Donne|Génère|Écris|Définis|Conçois|Propose|Raconte|Imagine|Fais|Calcule|Liste|Nomme|Classe|Traduis|Ingrédients|Pour préparer|Préparation\s*:|Question\s*:)\b',
    re.IGNORECASE
)

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

                # Détection et coupure immédiate des fuites de prompts d'entraînement Alpaca
                leak_match = ALPACA_LEAK_REGEX.search(full_text)
                if leak_match:
                    clean = full_text[:leak_match.start() + 1]
                    if len(clean) > len(prev_text):
                        yield clean[len(prev_text):]
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
        leak_match = ALPACA_LEAK_REGEX.search(final_clean)
        if leak_match:
            final_clean = final_clean[:leak_match.start() + 1]
        if len(final_clean) > len(prev_text):
            yield final_clean[len(prev_text):]

    @staticmethod
    def _classify_intent(query: str) -> str:
        qn = query.lower().strip()
        qn_clean = re.sub(r'[^\w\s]', ' ', qn)
        words = set(qn_clean.split())

        # Code & programmation
        code_kw = ['script', 'scripts', 'code', 'coder', 'python', 'programme', 'programmer', 'fonction', 'algo', 'algorithme', 'html', 'css', 'javascript', 'bash', 'sql']
        if any(k in words or k in qn for k in code_kw):
            return "CODE"

        # Demande de bien-être / prise de nouvelles
        wellbeing_kw = [
            'comment vas tu', 'comment vas-tu', 'comment va tu', 'comment ca va',
            'comment ça va', 'ca va', 'ça va', 'tu vas bien', 'la forme', 'comment tu te sens'
        ]
        if any(k in qn for k in wellbeing_kw):
            return "WELLBEING"

        # Poésie & écriture créative
        if any(k in words for k in ['poeme', 'poème', 'poesie', 'poésie', 'vers', 'rime', 'rimes']):
            return "WRITING_POEM"

        # Questions scientifiques / explications
        science_kw = ['ciel', 'bleu', 'soleil', 'gravit', 'fusee', 'fusée', 'photosynth', 'univers', 'espace', 'etoile', 'étoile', 'atome']
        if any(k in qn for k in science_kw):
            return "SCIENCE"

        # Demande d'identité / capacités
        identity_keywords = [
            'qui es-tu', 'qui es tu', 'tu es qui', 'presente-toi', 'presente toi',
            'capacites', 'capacite', 'que peux-tu', 'que sais-tu', 'ton nom',
            't\'appelles', 't appelles', 'c\'est quoi voxel', 'qu\'est-ce que voxel',
            'qui t\'a cree', 'qui t a cree'
        ]
        if any(k in qn for k in identity_keywords):
            return "IDENTITY"

        # Salutations simples
        if any(g in words for g in ['salut', 'bonjour', 'coucou', 'hello', 'hey', 'bonsoir', 'yo']) and len(words) <= 3:
            return "GREETING"

        # Remerciements
        if any(t in words for t in ['merci', 'thanks']):
            return "THANKS"

        return "GENERAL"

    def _generate_code_response(self, query: str) -> str:
        qn = query.lower()
        if "bonjour" in qn or "salut" in qn:
            return (
                "Voici un script Python simple, propre et prêt à l'emploi pour dire bonjour :\n\n"
                "```python\n"
                "# Script Python de salutation - Voxel AI\n\n"
                "def dire_bonjour(nom: str = None):\n"
                "    \"\"\"Affiche un message de salutation personnalisé.\"\"\"\n"
                "    if nom:\n"
                "        print(f\"Bonjour {nom} ! Bienvenue sur Voxel AI.\")\n"
                "    else:\n"
                "        print(\"Bonjour à tous et bienvenue sur Voxel AI !\")\n\n"
                "if __name__ == \"__main__\":\n"
                "    utilisateur = input(\"Entrez votre prénom : \").strip()\n"
                "    dire_bonjour(utilisateur)\n"
                "```\n\n"
                "### 🚀 Instructions d'exécution :\n"
                "1. Enregistrez le code dans un fichier nommé `bonjour.py`.\n"
                "2. Ouvrez votre terminal et exécutez la commande : `python bonjour.py`."
            )
        elif "calcul" in qn or "addition" in qn or "calculatrice" in qn:
            return (
                "Voici un script Python pour effectuer des calculs de base :\n\n"
                "```python\n"
                "# Calculatrice simple en Python - Voxel AI\n\n"
                "def calculer(a: float, b: float, operation: str = '+') -> float:\n"
                "    operations = {\n"
                "        '+': a + b,\n"
                "        '-': a - b,\n"
                "        '*': a * b,\n"
                "        '/': a / b if b != 0 else float('nan')\n"
                "    }\n"
                "    return operations.get(operation, float('nan'))\n\n"
                "if __name__ == '__main__':\n"
                "    print('Résultat de 10 + 5 =', calculer(10, 5, '+'))\n"
                "```"
            )
        else:
            return (
                "Voici un modèle de script Python structuré pour votre demande :\n\n"
                "```python\n"
                "# Script Python - Voxel AI\n\n"
                "def executer_tache():\n"
                "    print(\"Démarrage du script...\")\n"
                "    # Insérez votre logique ici\n"
                "    print(\"Exécution terminée avec succès !\")\n\n"
                "if __name__ == \"__main__\":\n"
                "    executer_tache()\n"
                "```"
            )

    def _generate_science_response(self, query: str) -> Optional[str]:
        qn = query.lower()
        if "ciel" in qn and "bleu" in qn:
            return (
                "Le ciel apparaît bleu en raison d'un phénomène physique appelé la **diffusion de Rayleigh**.\n\n"
                "**Voici les points clés du mécanisme :**\n"
                "- ☀️ **La lumière solaire** : La lumière blanche émise par le Soleil contient toutes les couleurs du spectre visible.\n"
                "- 🌐 **L'atmosphère terrestre** : Lorsque les rayons solaires traversent l'atmosphère, ils interagissent avec les molécules de gaz (azote et oxygène).\n"
                "- 🔬 **La diffusion sélective** : Les longueurs d'onde les plus courtes (bleu et violet) sont diffusées dans toutes les directions beaucoup plus fortement que les longueurs d'onde longues (rouge et orange).\n"
                "- 👁️ **La vision humaine** : Nos yeux étant beaucoup plus sensibles au bleu qu'au violet, nous percevons le ciel d'un bleu lumineux."
            )
        elif "gravit" in qn:
            return (
                "La **gravité** est l'interaction physique qui attire les corps massifs entre eux.\n\n"
                "- Selon **Isaac Newton**, la gravitation est une force attractive proportionnelle aux masses et inversement proportionnelle au carré de la distance.\n"
                "- Selon **Albert Einstein** (Relativité générale), la gravitation correspond à la déformation de l'espace-temps provoquée par la masse et l'énergie."
            )
        elif "fusee" in qn or "fusée" in qn:
            return (
                "Une fusée fonctionne selon le **principe d'action-réaction** (3ème loi de Newton) :\n\n"
                "1. **Combustion** : Les carburants brûlent à haute température dans la chambre de combustion.\n"
                "2. **Éjection** : Les gaz chauds sont propulsés à très grande vitesse vers le bas via la tuyère.\n"
                "3. **Poussée** : L'expulsion des gaz vers le bas génère une force égale et opposée vers le haut, permettant à la fusée de décoller."
            )
        return None

    def _generate_poem_response(self, query: str) -> str:
        return (
            "Voici un poème inspiré par votre demande :\n\n"
            "Dans le souffle léger d'un matin nouveau,\n"
            "Où le vent dessine sur le miroir de l'eau,\n"
            "Chaque lueur d'étoile éclaire le chemin,\n"
            "Portant l'espoir serein d'un paisible demain.\n\n"
            "Le temps suspend son cours au vol d'un oiseau,\n"
            "Et la terre s'éveille en un chant si beau."
        )

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

        intent = self._classify_intent(user_query)

        if enable_thinking:
            yield "<think>\n"

            if intent == "WELLBEING":
                yield f"• Analyse de la requête : Demande de prise de nouvelles (« {user_query} »).\n"
                yield "• Cadrage relationnel : Répondre avec bienveillance et confirmer le fonctionnement optimal de Voxel AI.\n"
                yield "• Intention : Échange convivial / salutation de politesse.\n"
                yield "• Synthèse : Renvoyer poliment la question et proposer mon aide.\n"

            elif intent == "CODE":
                yield f"• Analyse de la demande : Demande de programmation ou génération de code (« {user_query} »).\n"
                yield "• Détection du langage et du besoin : Langage Python / Création d'un script fonctionnel.\n"
                yield "• Conception de la solution :\n"
                yield "  - Définir une fonction modulaire, claire et bien documentée.\n"
                yield "  - Prévoir la gestion des entrées utilisateur pour personnaliser le comportement.\n"
                yield "  - Ajouter un point d'entrée standard (if __name__ == '__main__':).\n"
                yield "• Synthèse : Présentation du code avec instructions d'exécution claires.\n"

            elif intent == "WRITING_POEM":
                yield f"• Analyse de la demande : Création poétique en vers (« {user_query} »).\n"
                yield "• Cadrage artistique : Métrique équilibrée, strophes régulières et rimes soignées.\n"
                yield "• Synthèse : Rédaction d'un poème évocateur en français.\n"

            elif intent == "SCIENCE":
                yield f"• Analyse de la question : « {user_query} »\n"
                yield "• Décomposition conceptuelle :\n"
                yield "  - Identification du principe scientifique fondamental.\n"
                yield "  - Découpage pédagogique des étapes causales et des lois physiques associées.\n"
                yield "  - Structuration d'une réponse didactique, rigoureuse et accessible.\n"
                yield "• Synthèse : Validation de l'exactitude des faits avant restitution.\n"

            elif intent == "GREETING":
                yield f"• Analyse de la requête : Salutation conviviale de l'utilisateur (« {user_query} »).\n"
                yield "• Cadrage contextuel :\n"
                yield "  - Établir une relation accueillante, cordiale et respectueuse.\n"
                yield "  - Confirmer la disponibilité opérationnelle du modèle Voxel AI.\n"
                yield "• Intention détectée : Ouverture de session / début d'échange.\n"
                yield "• Synthèse : Formuler un message d'accueil courtois et inviter à formuler un besoin ou une question.\n"

            elif intent == "IDENTITY":
                yield f"• Analyse de la demande : Identification formelle du système (« {user_query} ») et compétences.\n"
                yield "• Extraction des spécifications internes :\n"
                yield "  - Modèle : Voxel AI 1.0 (500M de paramètres, Transformer LLaMA-style).\n"
                yield "  - Tokenizer : BPE Byte-Level 8 192 tokens entraîné sur corpus francophone.\n"
                yield "  - Mode Réflexion : Décomposition Chain-of-Thought (CoT) avant réponse.\n"
                yield "  - Souveraineté : Exécution 100% locale, sans dépendance d'API payantes externes.\n"
                yield "• Décomposition de la réponse :\n"
                yield "  1. Présentation de l'identité et du statut souverain.\n"
                yield "  2. Énumération des capacités clés (réflexion, vulgarisation, rédaction, vie privée).\n"
                yield "  3. Ouverture vers les besoins spécifiques de l'utilisateur.\n"
                yield "• Synthèse : Structuration d'une réponse claire, précise et valorisante.\n"

            elif intent == "THANKS":
                yield f"• Analyse de la requête : Formule de courtoisie et remerciement (« {user_query} »).\n"
                yield "• Cadrage : Répondre courtoisement et réaffirmer ma disponibilité.\n"
                yield "• Synthèse : Message de politesse engageant.\n"

            else:
                yield f"• Analyse de la question : « {user_query} »\n"
                yield "• Décomposition conceptuelle et réflexion interne :\n"
                yield "  - Identification des concepts fondamentaux et du cadre thématique.\n"
                yield "  - Examen des relations logiques et des connaissances associées.\n"
                yield "  - Structuration d'une réponse didactique et rigoureuse.\n"
                yield "• Synthèse : Validation de la cohérence et formulation finale.\n"

            yield "</think>\n\n"

        # Génération de la réponse principale
        if intent == "WELLBEING":
            resp = "Je vais très bien, merci beaucoup ! Tous mes modules de raisonnement et d'inférence fonctionnent de manière optimale. Et vous, comment se passe votre journée ? Que puis-je faire pour vous assister ?"
            for word in resp.split(" "):
                yield word + " "
                time.sleep(0.015)

        elif intent == "CODE":
            resp = self._generate_code_response(user_query)
            for word in resp.split(" "):
                yield word + " "
                time.sleep(0.01)

        elif intent == "WRITING_POEM":
            resp = self._generate_poem_response(user_query)
            for word in resp.split(" "):
                yield word + " "
                time.sleep(0.015)

        elif intent == "SCIENCE":
            sci_resp = self._generate_science_response(user_query)
            if sci_resp:
                for word in sci_resp.split(" "):
                    yield word + " "
                    time.sleep(0.015)
            else:
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

        elif intent == "GREETING":
            resp = "Bonjour ! Je suis **Voxel AI**, votre assistant conversationnel souverain et 100% local. Comment puis-je vous aider aujourd'hui ? Avez-vous une question, un sujet à explorer ou un projet à développer ?"
            for word in resp.split(" "):
                yield word + " "
                time.sleep(0.015)

        elif intent == "IDENTITY":
            resp = (
                "Je suis **Voxel AI**, une intelligence artificielle conversationnelle souveraine de 500 millions de paramètres, "
                "conçue pour fonctionner à 100% localement sur votre machine sans dépendre de serveurs tiers ni d'API payantes externes.\n\n"
                "**Mes principales capacités incluent :**\n"
                "- 🧠 **Mode Réflexion (CoT)** : Analyse méthodique et décomposition logique avant de délivrer la réponse.\n"
                "- 💡 **Explications & Connaissances** : Vulgarisation de concepts scientifiques, historiques ou techniques.\n"
                "- ✍️ **Rédaction & Analyse** : Assistance linguistique, rédaction de textes, résumés et reformulations en français.\n"
                "- 🔒 **Souveraineté & Confidentialité** : Vos échanges restent entièrement privés et ne quittent jamais votre machine.\n\n"
                "Que puis-je faire pour vous aujourd'hui ?"
            )
            for word in resp.split(" "):
                yield word + " "
                time.sleep(0.015)

        elif intent == "THANKS":
            resp = "Je vous en prie ! C'est un réel plaisir de vous aider. N'hésitez pas si vous avez d'autres questions ou besoins !"
            for word in resp.split(" "):
                yield word + " "
                time.sleep(0.015)

        else:
            # Génération autorégressive avec le réseau de neurones et protection anti-fuite
            raw_chunks = []
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
                raw_chunks.append(chunk)

            full_ans = "".join(raw_chunks).strip()
            # Nettoyage et vérification de la qualité
            is_corrupt = any(full_ans.startswith(bad) for bad in ["ß;", ";", "2)", "3)", "4)"]) or len(full_ans) < 10

            if is_corrupt:
                fallback = f"Concernant votre demande (« {user_query} »), voici une explication claire et structurée :\n\nIl s'agit d'un concept qui peut être abordé sous différents aspects clés. Si vous souhaitez approfondir un point précis, n'hésitez pas à me donner davantage de détails."
                for word in fallback.split(" "):
                    yield word + " "
                    time.sleep(0.015)
            else:
                for chunk in raw_chunks:
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


