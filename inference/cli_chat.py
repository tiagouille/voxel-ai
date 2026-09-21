import os
import sys
import argparse
from typing import List, Dict

# Support imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference.engine import VoxelInferenceEngine

BANNER = r"""
 __      __                _             _____ 
 \ \    / /               | |      /\   |_   _|
  \ \  / /___ __  _____  | |     /  \    | |  
   \ \/ // _ \\ \/ / _ \ | |    / /\ \   | |  
    \  /| (_) |>  <  __/ | |___/ ____ \ _| |_ 
     \/  \___//_/\_\___| |______/_/    \_\_____|  (~500M Parameters)
                                               
"""

def start_cli_chat(
    checkpoint_path: str,
    tokenizer_dir: str = "tokenizer",
    device: str = "cpu",
    dtype: str = "float16",
    system_prompt: str = "Tu es Voxel AI, une intelligence artificielle conversationnelle utile, claire et bienveillante."
):
    print(BANNER)
    print(f"[*] Chargement du moteur d'inférence ({device}, {dtype})...")
    
    if not os.path.exists(checkpoint_path):
        print(f"[!] ERREUR : Checkpoint introuvable dans {checkpoint_path}")
        print("[!] Veuillez spécifier un checkpoint valide avec --checkpoint <chemin>")
        return

    engine = VoxelInferenceEngine(
        checkpoint_path=checkpoint_path,
        tokenizer_dir=tokenizer_dir,
        device=device,
        dtype=dtype
    )

    print("\n" + "=" * 65)
    print(" Session de discussion interactive démarrée.")
    print(" Commandes disponibles :")
    print("   /clear   - Réinitialiser l'historique de la conversation")
    print("   /temp X  - Régler la température (ex: /temp 0.8)")
    print("   /exit    - Quitter la discussion")
    print("=" * 65 + "\n")

    history: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    temperature = 0.7
    top_p = 0.9

    while True:
        try:
            user_input = input("\n\033[1;36mVous :\033[0m ").strip()
            if not user_input:
                continue

            # Commandes
            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                print("\nAu revoir et à bientôt avec Voxel AI !")
                break
            elif user_input.lower() == "/clear":
                history = [{"role": "system", "content": system_prompt}]
                print("\n[✓] Historique de conversation réinitialisé.")
                continue
            elif user_input.lower().startswith("/temp "):
                try:
                    new_temp = float(user_input.split()[1])
                    temperature = max(0.01, min(2.0, new_temp))
                    print(f"\n[✓] Température ajustée à {temperature}")
                except Exception:
                    print("\n[!] Syntaxe : /temp 0.7")
                continue

            # Ajout du message utilisateur
            history.append({"role": "user", "content": user_input})

            # Formatage du prompt avec le template de chat
            prompt = engine.tokenizer.apply_chat_template(history, add_generation_prompt=True)

            print("\n\033[1;35mVoxel AI :\033[0m ", end="", flush=True)
            assistant_response = ""

            # Génération en streaming temps réel
            for chunk in engine.stream_generate(
                prompt=prompt,
                max_new_tokens=512,
                temperature=temperature,
                top_p=top_p
            ):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                assistant_response += chunk

            print()
            history.append({"role": "assistant", "content": assistant_response.strip()})

        except KeyboardInterrupt:
            print("\n\nSession interrompue par l'utilisateur.")
            break

def main():
    parser = argparse.ArgumentParser(description="Discussion CLI avec Voxel AI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/voxel_ai_final.pt", help="Chemin vers le checkpoint")
    parser.add_argument("--tokenizer", type=str, default="tokenizer", help="Dossier du tokenizer")
    parser.add_argument("--device", type=str, default="cpu", help="Périphérique ('cpu' ou 'cuda')")
    parser.add_argument("--dtype", type=str, default="float16", help="Précision ('float16', 'bfloat16' ou 'float32')")
    args = parser.parse_args()

    # Si voxel_ai_final.pt n'existe pas, tenter de trouver latest_checkpoint.pt
    if not os.path.exists(args.checkpoint):
        alt = "checkpoints/latest_checkpoint.pt"
        if os.path.exists(alt):
            args.checkpoint = alt

    start_cli_chat(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer,
        device=args.device,
        dtype=args.dtype
    )

if __name__ == "__main__":
    main()
