const elements = {
  composer: document.querySelector("#composer"),
  input: document.querySelector("#questionInput"),
  send: document.querySelector("#sendButton"),
  messages: document.querySelector("#messages"),
  empty: document.querySelector("#emptyState"),
  traceList: document.querySelector("#traceList"),
  traceEmpty: document.querySelector("#traceEmpty"),
  tracePanel: document.querySelector("#tracePanel"),
  traceToggle: document.querySelector("#traceToggle"),
  turnCount: document.querySelector("#turnCount"),
  serviceDot: document.querySelector("#serviceDot"),
  serviceText: document.querySelector("#serviceText"),
  newChat: document.querySelector("#newChat"),
};

const state = {
  running: false,
  answerElement: null,
  currentTurn: 0,
  controller: null,
};

function setRunning(running) {
  state.running = running;
  elements.input.disabled = running;
  elements.send.disabled = running;
  elements.send.title = running ? "正在回答" : "发送问题";
}

function resizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 140)}px`;
}

function addMessage(role, text, extraClass = "") {
  const article = document.createElement("article");
  article.className = `message ${role} ${extraClass}`.trim();

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "ME" : "SR";

  const body = document.createElement("div");
  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = role === "user" ? "你" : "SAPR-RAG";
  const content = document.createElement("p");
  content.className = "message-text";
  content.textContent = text;

  body.append(label, content);
  article.append(avatar, body);
  elements.messages.append(article);
  article.scrollIntoView({ behavior: "smooth", block: "end" });
  return content;
}

function addTrace(kind, title, content) {
  elements.traceEmpty.hidden = true;
  const item = document.createElement("section");
  item.className = `trace-item ${kind}`;
  const heading = document.createElement("p");
  heading.className = "trace-title";
  heading.textContent = title;
  const body = document.createElement("p");
  body.className = "trace-content";
  body.textContent = content;
  item.append(heading, body);
  elements.traceList.append(item);
  elements.tracePanel.scrollTop = elements.tracePanel.scrollHeight;
  return item;
}

function addDocuments(event) {
  elements.traceEmpty.hidden = true;
  const item = document.createElement("section");
  item.className = "trace-item documents";
  const heading = document.createElement("p");
  heading.className = "trace-title";
  heading.textContent = `第 ${event.turn + 1} 轮 · 候选文档`;
  const list = document.createElement("div");
  list.className = "document-list";

  event.documents.forEach((doc) => {
    const details = document.createElement("details");
    details.className = "document";
    const summary = document.createElement("summary");
    const title = document.createElement("span");
    title.textContent = doc.title || "Untitled";
    const score = document.createElement("span");
    score.className = "document-score";
    score.textContent = Number(doc.score).toFixed(3);
    const text = document.createElement("p");
    text.textContent = doc.text || "No excerpt available.";
    summary.append(title, score);
    details.append(summary, text);
    list.append(details);
  });

  item.append(heading, list);
  elements.traceList.append(item);
}

function setService(status, label) {
  elements.serviceDot.className = `service-dot ${status}`;
  elements.serviceText.textContent = label;
}

function renderEvent(event) {
  if (Number.isInteger(event.turn)) {
    state.currentTurn = Math.max(state.currentTurn, event.turn + 1);
    elements.turnCount.textContent = `${state.currentTurn} 轮`;
  }

  switch (event.type) {
    case "queued":
      state.answerElement = addMessage("assistant", "请求已进入队列，正在等待推理资源。", "");
      state.answerElement.classList.add("waiting");
      break;
    case "started":
      if (state.answerElement) {
        state.answerElement.textContent = "正在分析问题并规划检索路径…";
      }
      break;
    case "status": {
      const labels = {
        reason: "正在判断下一步行动",
        retrieve: "正在检索知识库",
        evidence: "正在筛选有效证据",
      };
      if (state.answerElement?.classList.contains("waiting")) {
        state.answerElement.textContent = labels[event.stage] || "正在处理";
      }
      break;
    }
    case "query":
      addTrace("query", `第 ${event.turn + 1} 轮 · 子查询`, event.query);
      break;
    case "documents":
      addDocuments(event);
      break;
    case "evidence":
      addTrace("evidence", `第 ${event.turn + 1} 轮 · 证据`, event.evidence);
      break;
    case "answer_start":
      if (state.answerElement) {
        state.answerElement.textContent = "";
        state.answerElement.classList.remove("waiting");
      }
      break;
    case "answer_delta":
      if (state.answerElement) {
        state.answerElement.textContent += event.delta;
        state.answerElement.parentElement.parentElement.scrollIntoView({
          behavior: "smooth",
          block: "end",
        });
      }
      break;
    case "done":
      if (state.answerElement && !state.answerElement.textContent.trim()) {
        state.answerElement.textContent = event.answer;
      }
      addTrace("answer", "回答完成", `共完成 ${event.turns} 轮推理`);
      setRunning(false);
      elements.input.focus();
      break;
    case "error": {
      const publicMessage = errorMessage(event.code);
      if (state.answerElement) {
        state.answerElement.textContent = publicMessage;
        state.answerElement.classList.remove("waiting");
      } else {
        addMessage("assistant", publicMessage, "error");
      }
      setRunning(false);
      break;
    }
    default:
      break;
  }
}

function errorMessage(code) {
  const messages = {
    invalid_agent_action: "模型没有生成可解析的检索或回答动作，请重新提问。",
    empty_answer: "模型生成了空回答，请重新提问。",
    max_turns_exceeded: "达到最大检索轮数，暂时无法形成可靠回答。",
    upstream_error: "模型或检索服务暂时不可用，请稍后重试。",
    internal_error: "服务出现异常，请稍后重试。",
  };
  return messages[code] || "本次请求未能完成，请稍后重试。";
}

async function consumeSSE(response) {
  if (!response.body) {
    throw new Error("Streaming response is unavailable.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";

    for (const frame of frames) {
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("\n");
      if (data) {
        renderEvent(JSON.parse(data));
      }
    }
    if (done) break;
  }
}

async function ask(question) {
  if (state.running) return;
  const normalized = question.trim();
  if (!normalized) return;

  elements.empty.classList.add("hidden");
  addMessage("user", normalized);
  elements.input.value = "";
  resizeInput();
  resetTrace();
  setRunning(true);
  state.controller = new AbortController();

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: normalized }),
      signal: state.controller.signal,
    });
    if (!response.ok) {
      let detail = `请求失败 (${response.status})`;
      try {
        const payload = await response.json();
        detail = payload.detail || detail;
      } catch (_error) {
        // Keep the HTTP status message when the body is not JSON.
      }
      throw new Error(detail);
    }
    await consumeSSE(response);
  } catch (error) {
    if (error.name !== "AbortError") {
      const target = state.answerElement || addMessage("assistant", "", "error");
      target.textContent = error.message || "网络连接中断，请稍后重试。";
      target.classList.remove("waiting");
    }
  } finally {
    setRunning(false);
    state.controller = null;
  }
}

function resetTrace() {
  elements.traceList.replaceChildren();
  elements.traceEmpty.hidden = false;
  elements.turnCount.textContent = "0 轮";
  state.currentTurn = 0;
}

function resetConversation() {
  state.controller?.abort();
  state.answerElement = null;
  elements.messages.replaceChildren();
  elements.empty.classList.remove("hidden");
  resetTrace();
  setRunning(false);
  elements.input.focus();
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    const data = await response.json();
    if (response.ok && data.status === "ok") {
      setService("online", "模型与检索服务在线");
    } else {
      setService("offline", "服务尚未完全就绪");
    }
  } catch (_error) {
    setService("offline", "无法连接推理服务");
  }
}

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(elements.input.value);
});

elements.input.addEventListener("input", resizeInput);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => ask(button.textContent));
});

elements.newChat.addEventListener("click", resetConversation);
elements.traceToggle.addEventListener("click", () => {
  const isOpen = elements.tracePanel.classList.toggle("open");
  elements.traceToggle.setAttribute("aria-expanded", String(isOpen));
});

resizeInput();
checkHealth();
