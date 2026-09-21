@echo off
chcp 65001 >nul
title Synchronisation GitHub - Voxel AI
echo ==========================================================
echo   🚀 ENVOI DES MODIFICATIONS VERS GITHUB (tiagouille/voxel-ai)
echo ==========================================================
echo.
cd /d "%~dp0"

git add .
set /p commit_msg="Description des modifications (ex: Amélioration du design) : "
if "%commit_msg%"=="" set commit_msg="Mise a jour Voxel AI"

git commit -m "%commit_msg%"
git push origin main

echo.
if %ERRORLEVEL% EQU 0 (
    echo [SUCCÈS] Vos modifications sont en ligne sur GitHub !
    echo Le serveur de votre ami peut maintenant faire un simple "git pull".
) else (
    echo [ATTENTION] Une erreur s'est produite lors de l'envoi vers GitHub.
)
echo.
pause
