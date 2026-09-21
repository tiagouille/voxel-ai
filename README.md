# 🧊 Voxel AI - Modèle Conversationnel Souverain & Interface Web

Interface Web temps réel et serveur d'inférence haute performance pour **Voxel AI**, un modèle de langage souverain de ~427 millions de paramètres entraîné à partir de zéro.

---

## ⚡ Caractéristiques
- **Architecture :** Decoder-Only Transformer (~427M paramètres)
- **Attention :** Grouped-Query Attention (GQA 24:8 / 3:1) avec Rotary Position Embeddings (RoPE)
- **Feed-Forward :** Activation SwiGLU ($d_{ff}=4096$) et normalisation RMSNorm
- **Backend :** FastAPI avec streaming temps réel (Server-Sent Events)
- **Frontend :** Interface sombre responsive personnalisée (Dark Mode néon)
- **Poids du modèle :** Hébergés sur [Hugging Face Hub (tiagouille/voxel-ai)](https://huggingface.co/tiagouille/voxel-ai)

---

## 🚀 Démarrage Rapide

### Sur Windows
Double-cliquez simplement sur :
👉 `DEMARRER_SERVEUR.bat`

### Sur Linux (Serveur dédié / VPS)
```bash
chmod +x start_server.sh
./start_server.sh
```

Le serveur va automatiquement :
1. Installer les dépendances Python nécessaires.
2. Télécharger les poids du modèle depuis [Hugging Face (tiagouille/voxel-ai)](https://huggingface.co/tiagouille/voxel-ai) s'ils ne sont pas déjà présents.
3. Démarrer le serveur web sur le port **8000** :
   👉 Ouvrez votre navigateur sur **`http://localhost:8000`** (ou l'adresse IP de votre serveur).

---

## 🎨 Personnalisation

- **Design / Couleurs :** Modifiez [`web/style.css`](web/style.css).
- **Interface & Textes :** Modifiez [`web/index.html`](web/index.html).
- **Logique Client :** Modifiez [`web/app.js`](web/app.js).
- **Paramètres IA (température, contexte, etc.) :** Modifiez [`config/model_config.json`](config/model_config.json).

Pour envoyer vos modifications en ligne d'un simple clic sur Windows, lancez `METTRE_A_JOUR_GITHUB.bat`.
