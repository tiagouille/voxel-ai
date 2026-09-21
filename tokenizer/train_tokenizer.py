import os
import sys
import glob
import argparse
from typing import List

# Support imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tokenizer.voxel_tokenizer import VoxelTokenizer
from config.config import VoxelConfig

def train_tokenizer(
    data_files: List[str],
    output_dir: str = "tokenizer",
    vocab_size: int = 32000,
    min_frequency: int = 2
) -> VoxelTokenizer:
    print(f"[*] Démarrage de l'entraînement du Tokenizer Voxel...")
    print(f"[*] Fichiers sources ({len(data_files)}) : {data_files[:3]}...")
    print(f"[*] Taille cible du vocabulaire : {vocab_size:,}")

    tokenizer = VoxelTokenizer.train_from_files(
        files=data_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency
    )

    print(f"[+] Tokenizer entraîné avec succès ! Vocab size effectif : {tokenizer.vocab_size:,}")
    tokenizer.save(output_dir)
    print(f"[+] Fichiers sauvegardés dans : {os.path.abspath(output_dir)}")
    return tokenizer

def main():
    parser = argparse.ArgumentParser(description="Entraînement du Tokenizer Voxel AI")
    parser.add_argument("--data_dir", type=str, default="dataset/data", help="Dossier contenant les données texte")
    parser.add_argument("--output_dir", type=str, default="tokenizer", help="Dossier d'export du tokenizer")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Taille du vocabulaire")
    parser.add_argument("--min_freq", type=int, default=2, help="Fréquence minimale d'apparition")
    args = parser.parse_args()

    # Trouver les fichiers .txt dans le dossier spécifié
    patterns = [os.path.join(args.data_dir, "*.txt"), os.path.join(args.data_dir, "**", "*.txt")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))

    if not files:
        print(f"[!] Aucun fichier .txt trouvé dans {args.data_dir}.")
        print("[!] Utilisation des données de démarrage synthétiques...")
        # Générer le jeu de données synthétique si nécessaire
        from dataset.synthetic_seed import generate_seed_corpus
        seed_path = os.path.join(args.data_dir, "seed_corpus.txt")
        generate_seed_corpus(seed_path)
        files = [seed_path]

    train_tokenizer(
        data_files=files,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq
    )

if __name__ == "__main__":
    main()
