from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import ollama

HOST = "127.0.0.1"
PORT = int(os.getenv("AGENT_PORT", "8765"))
MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CHAT_FILE = DATA_DIR / "chat_history.json"
CONTEXT_FILE = DATA_DIR / "context.md"
WORKSPACE_FILE = DATA_DIR / "workspace.json"
MAX_FILE_SIZE = 300_000
IGNORED_DIRS = {
    ".git",
    ".next",
    ".turbo",
    ".idea",
    ".vscode",
    "__pycache__",
    "node_modules",
    "vendor",
    "dist",
    "build",
}

BASE_SYSTEM_PROMPT = """
Voce e um assistente de engenharia de software.
Priorize respostas praticas, objetivas e tecnicamente corretas.
Considere arquitetura, legibilidade, manutencao, testes, seguranca e performance.
Explique decisoes com clareza e sugira proximos passos quando fizer sentido.
Responda sempre em portugues do Brasil.
Se houver arquivo aberto no editor, considere esse conteudo como a fonte principal de verdade.
""".strip()

FILE_EDIT_PROMPT = """
Voce atua como um agente de engenharia de software com permissao para editar o projeto.
Recebera:
- a workspace atual
- a lista resumida de arquivos do projeto
- o arquivo atualmente aberto, quando houver
- a instrucao do usuario

Sua resposta deve ser JSON valido, sem markdown e sem explicacoes extras, no formato:
{
  "assistant_message": "resumo curto do que foi feito",
  "operations": [
    {
      "path": "caminho/relativo/do/arquivo.ext",
      "content": "conteudo final completo do arquivo",
      "summary": "o que mudou nesse arquivo"
    }
  ]
}

Regras:
- use "operations": [] quando nenhuma alteracao de arquivo for necessaria
- quando editar ou criar arquivo, sempre devolva o conteudo final completo
- voce pode criar arquivos novos quando isso fizer sentido
- preserve ao maximo o que nao precisa mudar
- mantenha o estilo do codigo existente
- todos os caminhos devem ser relativos a workspace
- nao inclua cercas de codigo
""".strip()

INDEX_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agente de Engenharia</title>
  <style>
    :root {
      color-scheme: dark;
      --bg-top: #020403;
      --bg-mid: #050907;
      --bg-bottom: #000000;
      --panel: rgba(10, 13, 11, 0.88);
      --panel-strong: rgba(8, 10, 9, 0.96);
      --border: rgba(150, 255, 196, 0.12);
      --border-strong: rgba(78, 255, 149, 0.35);
      --text: #f5fff8;
      --muted: #97b39f;
      --accent: #22c55e;
      --accent-strong: #7dffae;
      --accent-soft: rgba(34, 197, 94, 0.12);
      --user: linear-gradient(135deg, rgba(255, 255, 255, 0.09), rgba(255, 255, 255, 0.04));
      --assistant: linear-gradient(135deg, rgba(34, 197, 94, 0.16), rgba(34, 197, 94, 0.07));
      --system: rgba(255, 255, 255, 0.04);
      --editor: #050806;
      --editor-line: rgba(255, 255, 255, 0.025);
      --danger-bg: rgba(255, 255, 255, 0.06);
      --danger-text: #ffffff;
      --shadow: 0 30px 80px rgba(0, 0, 0, 0.52);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      height: 100dvh;
      overflow: hidden;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(34, 197, 94, 0.18), transparent 24%),
        radial-gradient(circle at 82% 18%, rgba(255, 255, 255, 0.07), transparent 18%),
        radial-gradient(circle at bottom right, rgba(34, 197, 94, 0.1), transparent 22%),
        linear-gradient(160deg, var(--bg-top), var(--bg-mid) 48%, var(--bg-bottom));
      color: var(--text);
    }

    .shell {
      width: calc(100vw - 20px);
      max-width: 1880px;
      margin: 10px auto;
      height: calc(100dvh - 20px);
      display: grid;
      grid-template-columns: 320px minmax(560px, 1.05fr) minmax(760px, 1.35fr);
      grid-template-areas: "sidebar chat editor";
      gap: 12px;
      align-items: stretch;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
      overflow: hidden;
      height: 100%;
    }

    .chat-panel {
      grid-area: chat;
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-width: 0;
    }

    .sidebar {
      grid-area: sidebar;
      min-height: 0;
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      min-width: 0;
    }

    .editor-layout {
      grid-area: editor;
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      min-width: 0;
    }

    .header {
      padding: 22px 22px 14px;
      border-bottom: 1px solid var(--border);
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.015), transparent),
        linear-gradient(135deg, rgba(34, 197, 94, 0.09), rgba(255, 255, 255, 0.015));
    }

    .eyebrow {
      margin: 0 0 8px;
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent-strong);
      font-weight: 700;
    }

    h1, h2, h3 {
      margin: 0;
    }

    h1 {
      font-size: clamp(28px, 3vw, 38px);
      line-height: 1.05;
      font-weight: 700;
    }

    .subline {
      margin: 10px 0 0;
      color: var(--muted);
      line-height: 1.5;
    }

    .messages {
      padding: 18px 22px 10px;
      overflow-y: auto;
      display: grid;
      gap: 14px;
      align-content: start;
      min-height: 0;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.02), transparent 18%),
        linear-gradient(180deg, transparent, rgba(255, 255, 255, 0.015));
    }

    .message {
      max-width: min(88%, 760px);
      border-radius: 20px;
      padding: 14px 17px;
      line-height: 1.55;
      white-space: pre-wrap;
      border: 1px solid var(--border);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }

    .message.user {
      justify-self: end;
      background: var(--user);
    }

    .message.assistant {
      justify-self: start;
      background: var(--assistant);
    }

    .message.system {
      justify-self: center;
      max-width: 100%;
      background: var(--system);
      color: var(--muted);
      font-size: 14px;
    }

    .composer, .block {
      padding: 18px 22px 22px;
      border-top: 1px solid var(--border);
    }

    .block.soft {
      border-top: 1px solid rgba(129, 153, 191, 0.12);
    }

    textarea,
    input,
    button {
      font: inherit;
    }

    textarea,
    input[type="text"] {
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--border);
      padding: 12px 14px;
      background: rgba(3, 5, 4, 0.92);
      color: var(--text);
      outline: none;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }

    textarea::placeholder,
    input[type="text"]::placeholder {
      color: #6f8d77;
    }

    textarea:focus,
    input[type="text"]:focus {
      border-color: var(--border-strong);
      box-shadow:
        0 0 0 3px rgba(34, 197, 94, 0.14),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    #prompt {
      min-height: 118px;
      resize: none;
    }

    #contextInput {
      min-height: 180px;
      resize: vertical;
    }

    #fileContent {
      min-height: 0;
      height: 100%;
      resize: none;
      border: none;
      border-top: 1px solid var(--border);
      border-radius: 0;
      padding: 16px 18px 22px;
      background: var(--editor);
      font-family: Consolas, "Courier New", monospace;
      line-height: 1.55;
      background-image: linear-gradient(180deg, var(--editor-line), var(--editor-line));
      background-size: 100% 1.55em;
    }

    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 12px;
    }

    button {
      border: 1px solid transparent;
      border-radius: 999px;
      padding: 12px 18px;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease, border-color 120ms ease, background 120ms ease;
    }

    button:hover {
      transform: translateY(-1px);
    }

    button:disabled {
      opacity: 0.6;
      cursor: wait;
      transform: none;
    }

    .primary {
      background: linear-gradient(135deg, #16a34a, #22c55e);
      color: #031006;
      font-weight: 700;
      box-shadow: 0 10px 24px rgba(34, 197, 94, 0.2);
    }

    .secondary {
      background: var(--accent-soft);
      color: var(--accent-strong);
      font-weight: 700;
      border: 1px solid rgba(125, 255, 174, 0.16);
    }

    .danger {
      background: var(--danger-bg);
      color: var(--danger-text);
      font-weight: 700;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .meta {
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
      margin-top: 12px;
    }

    .status {
      min-height: 24px;
      margin-top: 10px;
      color: var(--accent-strong);
      font-size: 14px;
      font-weight: 600;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(34, 197, 94, 0.1);
      color: var(--accent-strong);
      font-size: 13px;
      font-weight: 700;
      width: fit-content;
      border: 1px solid rgba(125, 255, 174, 0.14);
    }

    .files {
      overflow-y: auto;
      padding: 12px 0;
      min-height: 0;
    }

    .file-item {
      display: block;
      width: 100%;
      text-align: left;
      border: none;
      background: transparent;
      color: var(--text);
      padding: 10px 22px;
      border-radius: 0;
      font-size: 14px;
      line-height: 1.4;
    }

    .file-item:hover {
      transform: none;
      background: rgba(34, 197, 94, 0.08);
    }

    .file-item.active {
      background: rgba(34, 197, 94, 0.14);
      color: var(--accent-strong);
      font-weight: 700;
      box-shadow: inset 3px 0 0 #3bff8b;
    }

    .muted {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }

    .editor-head {
      padding: 18px 22px;
      display: grid;
      gap: 8px;
      border-bottom: 1px solid var(--border);
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.015), transparent),
        linear-gradient(135deg, rgba(34, 197, 94, 0.06), rgba(255, 255, 255, 0.01));
    }

    .editor-path {
      font-size: 14px;
      color: var(--muted);
      word-break: break-all;
    }

    @media (max-width: 1260px) {
      body {
        height: auto;
        min-height: 100dvh;
        overflow: auto;
      }

      .shell {
        width: min(100%, calc(100vw - 12px));
        max-width: none;
        height: auto;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        grid-template-areas:
          "chat chat"
          "editor editor"
          "sidebar sidebar";
      }

      .chat-panel, .sidebar, .editor-layout {
        min-height: auto;
      }

      .messages {
        max-height: 52dvh;
      }

      #fileContent {
        min-height: 420px;
        height: auto;
      }

      .files {
        max-height: 320px;
      }
    }

    @media (max-width: 920px) {
      .shell {
        width: min(100%, calc(100vw - 12px));
        margin: 6px auto;
        height: auto;
        gap: 8px;
        grid-template-columns: 1fr;
        grid-template-areas:
          "chat"
          "editor"
          "sidebar";
      }

      .panel {
        border-radius: 18px;
      }

      .header,
      .composer,
      .block,
      .editor-head {
        padding-left: 16px;
        padding-right: 16px;
      }

      .messages {
        padding-left: 16px;
        padding-right: 16px;
        max-height: none;
      }

      .message {
        max-width: 100%;
      }

      .actions {
        gap: 8px;
      }

      .actions button {
        flex: 1 1 180px;
      }

      #prompt {
        min-height: 108px;
      }

      #contextInput {
        min-height: 150px;
      }

      #fileContent {
        min-height: 360px;
      }

      .files {
        min-height: 0;
        max-height: 240px;
      }
    }

    @media (max-width: 640px) {
      body {
        font-size: 15px;
      }

      .shell {
        width: calc(100vw - 8px);
        margin: 4px auto;
        gap: 6px;
      }

      .panel {
        border-radius: 16px;
      }

      .header,
      .composer,
      .block,
      .editor-head {
        padding: 14px;
      }

      .messages {
        padding: 14px;
        gap: 10px;
      }

      h1 {
        font-size: 26px;
      }

      .subline,
      .muted,
      .meta,
      .status,
      .editor-path {
        font-size: 13px;
      }

      .pill {
        font-size: 12px;
        padding: 7px 10px;
      }

      textarea,
      input[type="text"] {
        padding: 11px 12px;
      }

      button {
        width: 100%;
        padding: 11px 14px;
      }

      .actions {
        flex-direction: column;
      }

      .actions button {
        flex: none;
      }

      #prompt {
        min-height: 96px;
      }

      #contextInput {
        min-height: 132px;
      }

      #fileContent {
        min-height: 300px;
        padding: 14px;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <aside class="panel sidebar">
      <section class="header">
        <p class="eyebrow">ST Console</p>
        <h2>Workspace</h2>
        <div class="meta">
          <span>Modelo: <strong id="modelName">carregando...</strong></span>
          <span>Historico: <strong id="historyFile">-</strong></span>
          <span>Contexto: <strong id="contextFile">-</strong></span>
          <span>Workspace: <strong id="workspaceFile">-</strong></span>
        </div>
      </section>

      <section class="block soft">
        <h2>Projeto alvo</h2>
        <p class="muted">Informe a pasta do projeto que o agente pode explorar e editar.</p>
        <input id="workspacePath" type="text" placeholder="C:\\caminho\\do\\seu\\projeto">
        <div class="actions">
          <button id="saveWorkspaceBtn" class="secondary">Conectar pasta</button>
          <button id="refreshFilesBtn" class="secondary">Atualizar arquivos</button>
        </div>
        <div id="workspaceStatus" class="status"></div>
      </section>

      <section class="block soft">
        <h2>Contexto persistente</h2>
        <p class="muted">Guarde aqui stack, regras de arquitetura, objetivos do produto e preferencias tecnicas.</p>
        <textarea id="contextInput" placeholder="Exemplo: usamos FastAPI, PostgreSQL e testes com pytest..."></textarea>
        <div class="actions">
          <button id="saveContextBtn" class="secondary">Salvar contexto</button>
        </div>
        <div id="contextStatus" class="status"></div>
      </section>

      <section class="block soft">
        <h2>Arquivos do projeto</h2>
        <p class="muted">Abra um arquivo para o agente usar esse conteudo como contexto direto da conversa.</p>
        <div id="files" class="files"></div>
        <div id="filesStatus" class="status"></div>
      </section>
    </aside>

    <section class="panel chat-panel">
      <header class="header">
        <p class="eyebrow">ST Assistant</p>
        <h1>Agente de Engenharia</h1>
        <p class="subline">
          Converse com seu agente, mantenha memoria persistente e use o arquivo aberto como contexto real para analise e alteracao.
        </p>
      </header>

      <section id="messages" class="messages"></section>

      <section class="composer">
        <textarea id="prompt" placeholder="Descreva a tarefa, cole um erro, peca revisao de codigo ou solicite uma edicao no arquivo aberto..."></textarea>
        <div class="actions">
          <button id="sendBtn" class="primary">Enviar</button>
          <button id="clearBtn" class="danger">Limpar conversa</button>
        </div>
        <div id="chatStatus" class="status"></div>
      </section>
    </section>

    <section class="panel editor-layout">
      <div class="editor-head">
        <div class="pill">Editor ST</div>
        <h2>Arquivo ativo</h2>
        <div id="activeFileLabel" class="editor-path">Nenhum arquivo aberto.</div>
        <div class="actions">
          <button id="saveFileBtn" class="primary" disabled>Salvar arquivo</button>
          <button id="reloadFileBtn" class="secondary" disabled>Recarregar</button>
        </div>
        <div id="editorStatus" class="status"></div>
      </div>
      <textarea id="fileContent" placeholder="Selecione um arquivo do projeto para abrir aqui." disabled></textarea>
    </section>
  </main>

  <script>
    const messagesEl = document.getElementById("messages");
    const promptEl = document.getElementById("prompt");
    const contextEl = document.getElementById("contextInput");
    const workspacePathEl = document.getElementById("workspacePath");
    const filesEl = document.getElementById("files");
    const fileContentEl = document.getElementById("fileContent");
    const activeFileLabelEl = document.getElementById("activeFileLabel");
    const sendBtn = document.getElementById("sendBtn");
    const clearBtn = document.getElementById("clearBtn");
    const saveContextBtn = document.getElementById("saveContextBtn");
    const saveWorkspaceBtn = document.getElementById("saveWorkspaceBtn");
    const refreshFilesBtn = document.getElementById("refreshFilesBtn");
    const saveFileBtn = document.getElementById("saveFileBtn");
    const reloadFileBtn = document.getElementById("reloadFileBtn");
    const chatStatusEl = document.getElementById("chatStatus");
    const contextStatusEl = document.getElementById("contextStatus");
    const workspaceStatusEl = document.getElementById("workspaceStatus");
    const filesStatusEl = document.getElementById("filesStatus");
    const editorStatusEl = document.getElementById("editorStatus");
    const modelNameEl = document.getElementById("modelName");
    const historyFileEl = document.getElementById("historyFile");
    const contextFileEl = document.getElementById("contextFile");
    const workspaceFileEl = document.getElementById("workspaceFile");

    let activeFile = "";
    let files = [];

    function setText(el, message) {
      el.textContent = message || "";
    }

    function scrollToBottom() {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function renderMessages(messages) {
      messagesEl.innerHTML = "";

      if (!messages.length) {
        const empty = document.createElement("div");
        empty.className = "message system";
        empty.textContent = "A conversa comeca aqui. O contexto persistente sera aplicado automaticamente.";
        messagesEl.appendChild(empty);
        return;
      }

      for (const item of messages) {
        const bubble = document.createElement("div");
        bubble.className = `message ${item.role}`;
        bubble.textContent = item.content;
        messagesEl.appendChild(bubble);
      }

      scrollToBottom();
    }

    function renderFiles() {
      filesEl.innerHTML = "";

      if (!files.length) {
        const empty = document.createElement("div");
        empty.className = "muted";
        empty.style.padding = "0 22px 10px";
        empty.textContent = "Nenhum arquivo encontrado para a pasta selecionada.";
        filesEl.appendChild(empty);
        return;
      }

      for (const path of files) {
        const button = document.createElement("button");
        button.className = "file-item";
        if (path === activeFile) {
          button.classList.add("active");
        }
        button.textContent = path;
        button.addEventListener("click", () => openFile(path));
        filesEl.appendChild(button);
      }
    }

    function updateEditorState() {
      activeFileLabelEl.textContent = activeFile || "Nenhum arquivo aberto.";
      const enabled = Boolean(activeFile);
      fileContentEl.disabled = !enabled;
      saveFileBtn.disabled = !enabled;
      reloadFileBtn.disabled = !enabled;
      renderFiles();
    }

    async function loadState() {
      const response = await fetch("/api/state");
      const data = await response.json();

      renderMessages(data.messages);
      contextEl.value = data.context;
      workspacePathEl.value = data.workspace_path || "";
      modelNameEl.textContent = data.model;
      historyFileEl.textContent = data.history_file;
      contextFileEl.textContent = data.context_file;
      workspaceFileEl.textContent = data.workspace_file;
      files = data.files || [];
      renderFiles();
      updateEditorState();
    }

    async function sendMessage() {
      const prompt = promptEl.value.trim();
      if (!prompt) {
        setText(chatStatusEl, "Escreva uma mensagem antes de enviar.");
        return;
      }

      sendBtn.disabled = true;
      clearBtn.disabled = true;
      setText(chatStatusEl, "Consultando o Ollama...");

      try {
        const response = await fetch("/api/message", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt,
            active_file: activeFile,
            active_file_content: activeFile ? fileContentEl.value : ""
          })
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Falha ao enviar a mensagem.");
        }

        if (data.active_file) {
          activeFile = data.active_file;
          fileContentEl.value = data.active_file_content || "";
          updateEditorState();
        }

        if (data.files) {
          files = data.files;
          renderFiles();
        }

        promptEl.value = "";
        renderMessages(data.messages);
        setText(chatStatusEl, data.status_message || "Resposta recebida e historico salvo.");
        setText(editorStatusEl, data.editor_status || "");
      } catch (error) {
        setText(chatStatusEl, error.message);
      } finally {
        sendBtn.disabled = false;
        clearBtn.disabled = false;
      }
    }

    async function saveContext() {
      saveContextBtn.disabled = true;
      setText(contextStatusEl, "Salvando contexto...");

      try {
        const response = await fetch("/api/context", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ context: contextEl.value })
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Nao foi possivel salvar o contexto.");
        }

        setText(contextStatusEl, "Contexto salvo com sucesso.");
      } catch (error) {
        setText(contextStatusEl, error.message);
      } finally {
        saveContextBtn.disabled = false;
      }
    }

    async function saveWorkspace() {
      const workspace_path = workspacePathEl.value.trim();
      saveWorkspaceBtn.disabled = true;
      refreshFilesBtn.disabled = true;
      setText(workspaceStatusEl, "Conectando pasta...");

      try {
        const response = await fetch("/api/workspace", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ workspace_path })
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Nao foi possivel conectar a pasta.");
        }

        files = data.files || [];
        activeFile = "";
        fileContentEl.value = "";
        updateEditorState();
        setText(workspaceStatusEl, "Pasta conectada.");
        setText(filesStatusEl, `${files.length} arquivos disponiveis.`);
      } catch (error) {
        setText(workspaceStatusEl, error.message);
      } finally {
        saveWorkspaceBtn.disabled = false;
        refreshFilesBtn.disabled = false;
      }
    }

    async function refreshFiles() {
      setText(filesStatusEl, "Atualizando lista...");

      try {
        const response = await fetch("/api/files");
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Nao foi possivel listar os arquivos.");
        }

        files = data.files || [];
        renderFiles();
        setText(filesStatusEl, `${files.length} arquivos disponiveis.`);
      } catch (error) {
        setText(filesStatusEl, error.message);
      }
    }

    async function openFile(path) {
      setText(editorStatusEl, "Abrindo arquivo...");

      try {
        const response = await fetch(`/api/file?path=${encodeURIComponent(path)}`);
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Nao foi possivel abrir o arquivo.");
        }

        activeFile = data.path;
        fileContentEl.value = data.content;
        updateEditorState();
        setText(editorStatusEl, "Arquivo carregado.");
      } catch (error) {
        setText(editorStatusEl, error.message);
      }
    }

    async function saveFile() {
      if (!activeFile) {
        return;
      }

      saveFileBtn.disabled = true;
      reloadFileBtn.disabled = true;
      setText(editorStatusEl, "Salvando arquivo...");

      try {
        const response = await fetch("/api/file", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            path: activeFile,
            content: fileContentEl.value
          })
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Nao foi possivel salvar o arquivo.");
        }

        setText(editorStatusEl, "Arquivo salvo com sucesso.");
      } catch (error) {
        setText(editorStatusEl, error.message);
      } finally {
        saveFileBtn.disabled = false;
        reloadFileBtn.disabled = false;
      }
    }

    async function reloadFile() {
      if (!activeFile) {
        return;
      }
      await openFile(activeFile);
    }

    async function clearConversation() {
      if (!confirm("Deseja limpar o historico atual da conversa?")) {
        return;
      }

      clearBtn.disabled = true;
      sendBtn.disabled = true;
      setText(chatStatusEl, "Limpando historico...");

      try {
        const response = await fetch("/api/reset", { method: "POST" });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Nao foi possivel limpar o historico.");
        }

        renderMessages(data.messages);
        setText(chatStatusEl, "Historico limpo.");
      } catch (error) {
        setText(chatStatusEl, error.message);
      } finally {
        clearBtn.disabled = false;
        sendBtn.disabled = false;
      }
    }

    sendBtn.addEventListener("click", sendMessage);
    clearBtn.addEventListener("click", clearConversation);
    saveContextBtn.addEventListener("click", saveContext);
    saveWorkspaceBtn.addEventListener("click", saveWorkspace);
    refreshFilesBtn.addEventListener("click", refreshFiles);
    saveFileBtn.addEventListener("click", saveFile);
    reloadFileBtn.addEventListener("click", reloadFile);

    promptEl.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        sendMessage();
      }
    });

    loadState().catch((error) => {
      setText(chatStatusEl, error.message || "Nao foi possivel carregar o estado inicial.");
    });
  </script>
</body>
</html>
"""


class ChatStore:
    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not CONTEXT_FILE.exists():
            CONTEXT_FILE.write_text(
                "# Contexto persistente\n\n"
                "Descreva aqui o que o agente deve lembrar sempre.\n",
                encoding="utf-8",
            )
        if not CHAT_FILE.exists():
            self._write_json([])
        if not WORKSPACE_FILE.exists():
            self._write_workspace({"workspace_path": ""})
        self._lock = threading.Lock()

    def _write_json(self, messages: list[dict[str, str]]) -> None:
        payload = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "messages": messages,
        }
        CHAT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_workspace(self, payload: dict[str, str]) -> None:
        WORKSPACE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_messages(self) -> list[dict[str, str]]:
        try:
            raw = json.loads(CHAT_FILE.read_text(encoding="utf-8"))
            messages = raw.get("messages", [])
        except (json.JSONDecodeError, FileNotFoundError):
            messages = []
        return [self._normalize_message(item) for item in messages if self._is_valid_message(item)]

    def save_messages(self, messages: list[dict[str, str]]) -> None:
        with self._lock:
            self._write_json(messages)

    def read_context(self) -> str:
        return CONTEXT_FILE.read_text(encoding="utf-8").strip()

    def save_context(self, context: str) -> None:
        with self._lock:
            CONTEXT_FILE.write_text(context.strip() + "\n", encoding="utf-8")

    def reset_messages(self) -> list[dict[str, str]]:
        with self._lock:
            self._write_json([])
        return []

    def read_workspace_path(self) -> str:
        try:
            raw = json.loads(WORKSPACE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            raw = {"workspace_path": ""}
        return str(raw.get("workspace_path", "")).strip()

    def save_workspace_path(self, workspace_path: str) -> None:
        with self._lock:
            self._write_workspace({"workspace_path": workspace_path})

    @staticmethod
    def _is_valid_message(item: Any) -> bool:
        return isinstance(item, dict) and item.get("role") in {"user", "assistant", "system"} and isinstance(item.get("content"), str)

    @staticmethod
    def _normalize_message(item: dict[str, str]) -> dict[str, str]:
        return {"role": item["role"], "content": item["content"].strip()}


class WorkspaceManager:
    def list_files(self, workspace_path: str) -> list[str]:
        root = self._resolve_root(workspace_path)
        files: list[str] = []

        for path in root.rglob("*"):
            if path.is_dir() and path.name in IGNORED_DIRS:
                continue
            if not path.is_file():
                continue
            if self._is_ignored(path):
                continue
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                continue
            files.append(relative)

        files.sort()
        return files

    def read_file(self, workspace_path: str, relative_path: str) -> dict[str, str]:
        file_path = self._resolve_file(workspace_path, relative_path)
        if file_path.stat().st_size > MAX_FILE_SIZE:
            raise ValueError("Arquivo grande demais para abrir no editor local.")
        if not self._is_text_file(file_path):
            raise ValueError("O arquivo selecionado nao parece ser texto editavel.")
        return {
            "path": file_path.relative_to(self._resolve_root(workspace_path)).as_posix(),
            "content": file_path.read_text(encoding="utf-8"),
        }

    def write_file(self, workspace_path: str, relative_path: str, content: str) -> dict[str, str]:
        file_path = self._resolve_file(workspace_path, relative_path, allow_create=True)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return {"path": file_path.relative_to(self._resolve_root(workspace_path)).as_posix()}

    def safe_file_list(self, workspace_path: str, limit: int = 200) -> list[str]:
        return self.list_files(workspace_path)[:limit]

    def _resolve_root(self, workspace_path: str) -> Path:
        if not workspace_path.strip():
            raise ValueError("Informe uma pasta de projeto antes de listar arquivos.")
        root = Path(workspace_path).expanduser().resolve()
        if not root.exists():
            raise ValueError("A pasta informada nao existe.")
        if not root.is_dir():
            raise ValueError("O caminho informado nao e uma pasta.")
        return root

    def _resolve_file(self, workspace_path: str, relative_path: str, *, allow_create: bool = False) -> Path:
        root = self._resolve_root(workspace_path)
        target = (root / relative_path).resolve()
        if not self._is_within_root(root, target):
            raise ValueError("O arquivo precisa estar dentro da pasta do projeto selecionada.")
        if not allow_create and not target.exists():
            raise ValueError("O arquivo solicitado nao existe.")
        if target.exists() and not target.is_file():
            raise ValueError("O caminho selecionado nao e um arquivo.")
        return target

    @staticmethod
    def _is_within_root(root: Path, target: Path) -> bool:
        try:
            target.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_ignored(path: Path) -> bool:
        return any(part in IGNORED_DIRS for part in path.parts)

    @staticmethod
    def _is_text_file(path: Path) -> bool:
        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type and mime_type.startswith("text/"):
            return True
        try:
            sample = path.read_bytes()[:2048]
        except OSError:
            return False
        return b"\x00" not in sample


store = ChatStore()
workspace = WorkspaceManager()


def build_model_messages(
    chat_messages: list[dict[str, str]],
    active_file: str = "",
    active_file_content: str = "",
) -> list[dict[str, str]]:
    context = store.read_context()
    system_parts = [BASE_SYSTEM_PROMPT]

    if context:
        system_parts.append("Contexto persistente do usuario:\n" + context)

    workspace_path = store.read_workspace_path()
    if workspace_path:
        system_parts.append(f"Workspace atual configurada: {workspace_path}")

    if active_file and active_file_content:
        system_parts.append(
            "Arquivo atualmente aberto no editor:\n"
            f"Caminho relativo: {active_file}\n"
            "Conteudo atual:\n"
            f"{active_file_content}"
        )

    return [{"role": "system", "content": "\n\n".join(system_parts)}] + chat_messages


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not json_match:
            raise ValueError("A IA nao retornou JSON valido.")
        payload = json.loads(json_match.group(0))

    if not isinstance(payload, dict):
        raise ValueError("A resposta da IA nao veio no formato esperado.")
    return payload


def _apply_agent_operations(
    workspace_path: str,
    operations: list[dict[str, Any]],
    fallback_active_file: str = "",
) -> dict[str, Any]:
    changed_files: list[dict[str, str]] = []
    active_file = fallback_active_file
    active_file_content = ""

    for operation in operations:
        if not isinstance(operation, dict):
            continue
        relative_path = str(operation.get("path", "")).strip()
        content = operation.get("content", "")
        summary = str(operation.get("summary", "")).strip()

        if not relative_path or not isinstance(content, str):
            continue

        result = workspace.write_file(workspace_path, relative_path, content.rstrip("\n") + "\n")
        changed_files.append({"path": result["path"], "summary": summary})

        if result["path"] == fallback_active_file or not active_file:
            active_file = result["path"]
            active_file_content = content.rstrip("\n") + "\n"

    return {
        "changed_files": changed_files,
        "active_file": active_file,
        "active_file_content": active_file_content,
    }


def ask_agent(prompt: str, active_file: str = "", active_file_content: str = "") -> dict[str, Any]:
    workspace_path = store.read_workspace_path()

    chat_messages = store.read_messages()
    chat_messages.append({"role": "user", "content": prompt.strip()})

    if not workspace_path:
        model_messages = build_model_messages(chat_messages, active_file, active_file_content)
        response: Any = ollama.chat(model=MODEL, messages=model_messages)
        answer = response["message"]["content"].strip()
        chat_messages.append({"role": "assistant", "content": answer})
        store.save_messages(chat_messages)
        return {
            "answer": answer,
            "messages": chat_messages,
            "changed_files": [],
            "active_file": active_file,
            "active_file_content": active_file_content,
        }

    project_files = workspace.safe_file_list(workspace_path)
    summarized_tree = "\n".join(f"- {path}" for path in project_files) or "- nenhum arquivo encontrado"
    instruction = (
        f"Workspace: {workspace_path}\n\n"
        "Arquivos conhecidos do projeto:\n"
        f"{summarized_tree}\n\n"
        f"Arquivo aberto: {active_file or 'nenhum'}\n\n"
        "Conteudo do arquivo aberto:\n"
        f"{active_file_content or '[nenhum arquivo aberto]'}\n\n"
        "Instrucao do usuario:\n"
        f"{prompt}"
    )

    messages = [
        {"role": "system", "content": FILE_EDIT_PROMPT},
        {"role": "user", "content": instruction},
    ]

    response: Any = ollama.chat(model=MODEL, messages=messages)
    raw_reply = response["message"]["content"]

    try:
        payload = _extract_json_payload(raw_reply)
    except ValueError:
        chat_messages.append({"role": "assistant", "content": raw_reply.strip()})
        store.save_messages(chat_messages)
        return {
            "answer": raw_reply.strip(),
            "messages": chat_messages,
            "changed_files": [],
            "active_file": active_file,
            "active_file_content": active_file_content,
            "status_message": "Resposta recebida sem alteracoes automaticas.",
            "editor_status": "",
            "files": workspace.list_files(workspace_path),
        }

    assistant_message = str(payload.get("assistant_message", "")).strip()
    operations = payload.get("operations", [])
    if not isinstance(operations, list):
        operations = []

    applied = _apply_agent_operations(workspace_path, operations, active_file)
    changed_files = applied["changed_files"]
    new_active_file = applied["active_file"] or active_file
    new_active_file_content = applied["active_file_content"] or active_file_content

    if not assistant_message:
        if changed_files:
            assistant_message = "Apliquei alteracoes no projeto com base no seu pedido."
        else:
            assistant_message = "Analisei sua solicitacao e, por enquanto, nao foi necessario alterar arquivos."

    chat_messages.append({"role": "assistant", "content": assistant_message})
    store.save_messages(chat_messages)

    status_message = "Resposta recebida."
    editor_status = ""
    if changed_files:
        status_message = f"Alteracoes aplicadas em {len(changed_files)} arquivo(s)."
        editor_status = "Arquivos atualizados automaticamente pela IA."

    return {
        "answer": assistant_message,
        "messages": chat_messages,
        "changed_files": changed_files,
        "active_file": new_active_file,
        "active_file_content": new_active_file_content,
        "status_message": status_message,
        "editor_status": editor_status,
        "files": workspace.list_files(workspace_path),
    }


class AgentHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return

        if parsed.path == "/api/state":
            workspace_path = store.read_workspace_path()
            files = self._safe_list_files(workspace_path)
            self._send_json(
                {
                    "messages": store.read_messages(),
                    "context": store.read_context(),
                    "model": MODEL,
                    "history_file": str(CHAT_FILE.name),
                    "context_file": str(CONTEXT_FILE.name),
                    "workspace_file": str(WORKSPACE_FILE.name),
                    "workspace_path": workspace_path,
                    "files": files,
                }
            )
            return

        if parsed.path == "/api/files":
            workspace_path = store.read_workspace_path()
            if not workspace_path:
                self._send_json({"files": []})
                return
            try:
                files = workspace.list_files(workspace_path)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"files": files})
            return

        if parsed.path == "/api/file":
            params = parse_qs(parsed.query)
            relative_path = params.get("path", [""])[0]
            try:
                payload = workspace.read_file(store.read_workspace_path(), relative_path)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)
            return

        self._send_json({"error": "Rota nao encontrada."}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/api/message":
            payload = self._read_json_body()
            prompt = str(payload.get("prompt", "")).strip()
            active_file = str(payload.get("active_file", "")).strip()
            active_file_content = str(payload.get("active_file_content", ""))

            if not prompt:
                self._send_json({"error": "Envie uma mensagem valida."}, status=HTTPStatus.BAD_REQUEST)
                return

            try:
                result = ask_agent(prompt, active_file, active_file_content)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json(
                    {
                        "error": (
                            "Falha ao consultar o Ollama. Verifique se o servidor esta ativo "
                            f"e se o modelo '{MODEL}' esta disponivel. Detalhe: {exc}"
                        )
                    },
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return

            self._send_json(result)
            return

        if self.path == "/api/context":
            payload = self._read_json_body()
            context = str(payload.get("context", ""))
            store.save_context(context)
            self._send_json({"ok": True})
            return

        if self.path == "/api/workspace":
            payload = self._read_json_body()
            workspace_path = str(payload.get("workspace_path", "")).strip()
            try:
                files = workspace.list_files(workspace_path) if workspace_path else []
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            store.save_workspace_path(workspace_path)
            self._send_json({"ok": True, "files": files})
            return

        if self.path == "/api/file":
            payload = self._read_json_body()
            relative_path = str(payload.get("path", "")).strip()
            content = str(payload.get("content", ""))
            try:
                result = workspace.write_file(store.read_workspace_path(), relative_path, content)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True, **result})
            return

        if self.path == "/api/reset":
            messages = store.reset_messages()
            self._send_json({"ok": True, "messages": messages})
            return

        self._send_json({"error": "Rota nao encontrada."}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _safe_list_files(self, workspace_path: str) -> list[str]:
        if not workspace_path:
            return []
        try:
            return workspace.list_files(workspace_path)
        except ValueError:
            return []

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}

        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    server = ThreadingHTTPServer((HOST, PORT), AgentHandler)

    print(f"Agente local disponivel em {url}")
    print(f"Modelo ativo: {MODEL}")
    print(f"Historico salvo em: {CHAT_FILE}")
    print(f"Contexto persistente em: {CONTEXT_FILE}")
    print(f"Workspace salva em: {WORKSPACE_FILE}")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
