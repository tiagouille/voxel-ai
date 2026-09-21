@echo off
chcp 65001 >nul
title Serveur Web Voxel AI
echo ==========================================================
echo   🚀 DÉMARRAGE DU SERVEUR VOXEL AI (FASTAPI + INTERFACE)
echo ==========================================================
echo.
cd /d "%~dp0"

echo [*] Vérification des dépendances...
py -m pip install -r requirements.txt

echo.
echo [*] Lancement du serveur Web et de l'IA sur le port 8000...
echo     Interface accessible sur : http://localhost:8000
echo.
py server/app.py --port 8000 --host 0.0.0.0
pause
