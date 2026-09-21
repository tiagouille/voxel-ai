#!/usr/bin/env bash
# Script de lancement automatique pour Linux (Serveur VPS / Dédié)
set -e

echo "=========================================================="
echo "  🚀 DÉMARRAGE DU SERVEUR VOXEL AI (FASTAPI + INTERFACE)"
echo "=========================================================="

cd "$(dirname "$0")"

# Vérification de l'environnement virtuel Python
if [ ! -d "venv" ]; then
    echo "[*] Création de l'environnement virtuel Python..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "[*] Lancement du serveur Web et de l'IA sur le port 8000..."
echo "    Interface accessible sur : http://0.0.0.0:8000"
echo ""

python3 server/app.py --port 8000 --host 0.0.0.0
