FROM python:3.11-slim

WORKDIR /app

# Dépendances système légères
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie de tout le projet Voxel AI (y compris les poids et le web)
COPY . .

# Port officiel Hugging Face Spaces
EXPOSE 7860

# Lancement du serveur FastAPI sur le port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
