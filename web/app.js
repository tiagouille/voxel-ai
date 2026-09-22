document.addEventListener("DOMContentLoaded", () => {
  // Éléments du DOM
  const chatContainer = document.getElementById("chatContainer");
  const messagesList = document.getElementById("messagesList");
  const welcomeCard = document.getElementById("welcomeCard");
  const userInput = document.getElementById("userInput");
  const btnSend = document.getElementById("btnSend");
  const btnClear = document.getElementById("btnClear");
  const btnSettings = document.getElementById("btnSettings");
  const btnCloseSettings = document.getElementById("btnCloseSettings");
  const settingsModal = document.getElementById("settingsModal");
  const typingIndicator = document.getElementById("typingIndicator");
  const statusIndicator = document.getElementById("statusIndicator");
  const statusText = document.getElementById("statusText");

  // Contrôles de paramètres
  const tempRange = document.getElementById("tempRange");
  const tempVal = document.getElementById("tempVal");
  const topPRange = document.getElementById("topPRange");
  const topPVal = document.getElementById("topPVal");
  const maxTokensRange = document.getElementById("maxTokensRange");
  const maxTokensVal = document.getElementById("maxTokensVal");

  // État de l'application
  let conversationHistory = [
    { role: "system", content: "Tu es Voxel AI, une intelligence artificielle conversationnelle utile, claire et bienveillante." }
  ];
  let isGenerating = false;

  // Mode Réflexion (DeepThink)
  const btnThinkToggle = document.getElementById("btnThinkToggle");
  let enableThinking = true;

  if (btnThinkToggle) {
    btnThinkToggle.addEventListener("click", () => {
      enableThinking = !enableThinking;
      btnThinkToggle.classList.toggle("active", enableThinking);
      const textSpan = btnThinkToggle.querySelector(".think-text");
      if (textSpan) {
        textSpan.textContent = enableThinking ? "Mode Réflexion" : "Mode Direct";
      }
    });
  }

  // Initialisation des sliders
  tempRange.addEventListener("input", (e) => tempVal.textContent = e.target.value);
  topPRange.addEventListener("input", (e) => topPVal.textContent = parseFloat(e.target.value).toFixed(2));
  maxTokensRange.addEventListener("input", (e) => maxTokensVal.textContent = e.target.value);

  // Gestion de la modale de paramètres
  btnSettings.addEventListener("click", () => settingsModal.style.display = "flex");
  btnCloseSettings.addEventListener("click", () => settingsModal.style.display = "none");
  settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) settingsModal.style.display = "none";
  });

  // Vérification de la santé du serveur
  async function checkServerHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        statusIndicator.classList.add("online");
        if (data.model_loaded) {
          statusText.textContent = `En ligne (${data.device.toUpperCase()})`;
        } else {
          statusText.textContent = "Serveur prêt (En attente de modèle)";
        }
      } else {
        statusIndicator.classList.remove("online");
        statusText.textContent = "Erreur serveur";
      }
    } catch (e) {
      statusIndicator.classList.remove("online");
      statusText.textContent = "Déconnecté";
    }
  }

  checkServerHealth();
  setInterval(checkServerHealth, 10000);

  // Redimensionnement automatique du champ de texte
  userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 140) + "px";
    btnSend.disabled = userInput.value.trim().length === 0 || isGenerating;
  });

  // Touche Entrée pour envoyer (Maj+Entrée pour saut de ligne)
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!btnSend.disabled) sendMessage();
    }
  });

  btnSend.addEventListener("click", () => {
    if (!btnSend.disabled) sendMessage();
  });

  // Chips de questions rapides
  document.querySelectorAll(".prompt-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const prompt = btn.getAttribute("data-prompt");
      if (prompt) {
        userInput.value = prompt;
        userInput.dispatchEvent(new Event("input"));
        sendMessage();
      }
    });
  });

  // Réinitialiser la conversation
  btnClear.addEventListener("click", () => {
    if (confirm("Voulez-vous effacer l'historique de cette discussion ?")) {
      conversationHistory = [
        { role: "system", content: "Tu es Voxel AI, une intelligence artificielle conversationnelle utile, claire et bienveillante." }
      ];
      messagesList.innerHTML = "";
      welcomeCard.style.display = "block";
    }
  });

  // Envoi d'un message
  async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || isGenerating) return;

    // Masquer l'écran d'accueil dès le premier message
    welcomeCard.style.display = "none";

    // Ajouter le message utilisateur à l'UI et à l'historique
    appendMessage("user", text);
    conversationHistory.push({ role: "user", content: text });

    userInput.value = "";
    userInput.style.height = "auto";
    btnSend.disabled = true;
    isGenerating = true;

    // Afficher l'indicateur de génération
    typingIndicator.style.display = "flex";
    scrollToBottom();

    // Créer la bulle de réponse de l'assistant vide
    const botMessageRow = document.createElement("div");
    botMessageRow.className = "message-row assistant";
    botMessageRow.innerHTML = `
      <div class="voxel-avatar bot">V</div>
      <div class="message-bubble" id="currentBotBubble"></div>
    `;
    messagesList.appendChild(botMessageRow);
    const botBubble = botMessageRow.querySelector("#currentBotBubble");

    try {
      const payload = {
        messages: conversationHistory,
        temperature: parseFloat(tempRange.value),
        top_p: parseFloat(topPRange.value),
        max_tokens: parseInt(maxTokensRange.value),
        enable_thinking: enableThinking
      };

      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Erreur ${response.status}: ${response.statusText}`);
      }

      // Masquer l'indicateur de génération dès que le flux démarre
      typingIndicator.style.display = "none";

      // Lecture du flux SSE
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let fullAssistantText = "";
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); // Conserver le fragment incomplet

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.slice(6).trim();
            if (dataStr === "[DONE]") {
              break;
            }
            try {
              const dataObj = JSON.parse(dataStr);
              if (dataObj.token) {
                fullAssistantText += dataObj.token;
                renderAssistantBubble(botBubble, fullAssistantText);
                scrollToBottom();
              }
            } catch (err) {
              console.warn("Échec du parsing JSON du chunk :", dataStr);
            }
          }
        }
      }

      // Enregistrer la réponse dans l'historique
      conversationHistory.push({ role: "assistant", content: fullAssistantText });

    } catch (err) {
      typingIndicator.style.display = "none";
      botBubble.innerHTML = `<span style="color: #ef4444;">Erreur lors de la génération : ${err.message}</span>`;
    } finally {
      isGenerating = false;
      btnSend.disabled = userInput.value.trim().length === 0;
      botBubble.removeAttribute("id");
      scrollToBottom();
    }
  }

  function appendMessage(role, text) {
    const row = document.createElement("div");
    row.className = `message-row ${role}`;
    const avatar = role === "user" 
      ? '<div class="voxel-avatar user-avatar">U</div>' 
      : '<div class="voxel-avatar bot">V</div>';

    row.innerHTML = `
      ${avatar}
      <div class="message-bubble">${escapeHtml(text)}</div>
    `;
    messagesList.appendChild(row);
    scrollToBottom();
  }

  function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
    const viewport = document.querySelector(".chat-viewport");
    if (viewport) {
      viewport.scrollTop = viewport.scrollHeight;
    }
  }

  function renderAssistantBubble(bubble, text) {
    if (text.includes("<think>")) {
      const parts = text.split("<think>");
      const afterThink = parts[1] || "";
      let thinkingText = "";
      let answerText = "";

      if (afterThink.includes("</think>")) {
        const thinkParts = afterThink.split("</think>");
        thinkingText = thinkParts[0].trim();
        answerText = thinkParts.slice(1).join("</think>").trimStart();
      } else {
        thinkingText = afterThink.trim();
      }

      const isThinkingDone = afterThink.includes("</think>");

      let thinkingBlock = bubble.querySelector(".thinking-block");
      if (!thinkingBlock) {
        thinkingBlock = document.createElement("div");
        thinkingBlock.className = "thinking-block";
        thinkingBlock.innerHTML = `
          <div class="thinking-header">
            <span class="thinking-icon">💭</span>
            <span class="thinking-title">Réflexion de Voxel AI...</span>
            <span class="thinking-arrow">▾</span>
          </div>
          <div class="thinking-content"></div>
        `;
        thinkingBlock.querySelector(".thinking-header").addEventListener("click", () => {
          thinkingBlock.classList.toggle("collapsed");
        });
        bubble.innerHTML = "";
        bubble.appendChild(thinkingBlock);

        const answerDiv = document.createElement("div");
        answerDiv.className = "answer-text";
        bubble.appendChild(answerDiv);
      }

      const contentEl = thinkingBlock.querySelector(".thinking-content");
      const titleEl = thinkingBlock.querySelector(".thinking-title");
      contentEl.textContent = thinkingText;

      if (isThinkingDone) {
        titleEl.textContent = "Réflexion terminée (cliquer pour afficher)";
        if (!thinkingBlock.dataset.autoCollapsed && answerText.length > 5) {
          thinkingBlock.classList.add("collapsed");
          thinkingBlock.dataset.autoCollapsed = "true";
        }
      } else {
        titleEl.textContent = "Réflexion en cours...";
        thinkingBlock.classList.remove("collapsed");
      }

      const answerEl = bubble.querySelector(".answer-text");
      if (answerEl) {
        answerEl.innerHTML = formatMarkdown(answerText);
      }
    } else {
      bubble.innerHTML = formatMarkdown(text);
    }
  }

  function formatMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);
    // Gras : **texte**
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Puces : - élément
    html = html.replace(/(?:^|\n)- (.*?)(?=\n|$)/g, '<br>• $1');
    // Sauts de ligne
    html = html.replace(/\n/g, '<br>');
    return html;
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
