function setText(el, text) {
  el.textContent = text || "";
}

function renderNotes(targetEl, data, level) {
  const container = targetEl;
  container.innerHTML = "";

  if (!data || typeof data !== "object") {
    container.textContent = "(No notes returned.)";
    return;
  }

  const summary = document.createElement("div");
  summary.className = "blockquote";
  summary.textContent = data.summary || "(no summary)";
  container.appendChild(summary);

  if (Array.isArray(data.bullets) && data.bullets.length) {
    const h = document.createElement("div");
    h.className = "section-title";
    h.style.fontSize = "0.85rem";
    h.textContent = "Key Points";
    container.appendChild(h);

    const ul = document.createElement("ul");
    data.bullets.forEach((b) => {
      const li = document.createElement("li");
      li.textContent = b;
      ul.appendChild(li);
    });
    container.appendChild(ul);
  }

  if (Array.isArray(data.definitions) && data.definitions.length) {
    const h = document.createElement("div");
    h.className = "section-title";
    h.style.fontSize = "0.85rem";
    h.textContent = "Definitions";
    container.appendChild(h);

    const ul = document.createElement("ul");
    data.definitions.forEach((d) => {
      const li = document.createElement("li");
      li.textContent = d;
      ul.appendChild(li);
    });
    container.appendChild(ul);
  }

  if (level === "beginner" && data.example) {
    const h = document.createElement("div");
    h.className = "section-title";
    h.style.fontSize = "0.85rem";
    h.textContent = "Example";
    container.appendChild(h);

    const ex = document.createElement("div");
    ex.className = "blockquote";
    ex.textContent = data.example;
    container.appendChild(ex);
  }

  if ((level === "intermediate" || level === "advanced") && Array.isArray(data.examples) && data.examples.length) {
    const h = document.createElement("div");
    h.className = "section-title";
    h.style.fontSize = "0.85rem";
    h.textContent = "Examples";
    container.appendChild(h);

    const ul = document.createElement("ul");
    data.examples.forEach((ex) => {
      const li = document.createElement("li");
      li.textContent = ex;
      ul.appendChild(li);
    });
    container.appendChild(ul);
  }

  const quizKey = data.quick_quiz ? "quick_quiz" : "quiz";
  if (Array.isArray(data[quizKey]) && data[quizKey].length) {
    const h = document.createElement("div");
    h.className = "section-title";
    h.style.fontSize = "0.85rem";
    h.textContent = "Quiz";
    container.appendChild(h);

    data[quizKey].forEach((qa, idx) => {
      const div = document.createElement("div");
      div.className = "quiz-item";
      const q = document.createElement("div");
      q.className = "quiz-q";
      q.textContent = "Q" + (idx + 1) + ": " + (qa.q || "");
      const a = document.createElement("div");
      a.className = "quiz-a";
      a.textContent = "A" + (idx + 1) + ": " + (qa.a || "");
      div.appendChild(q);
      div.appendChild(a);
      container.appendChild(div);
    });
  }
}

function renderEducationalContent(data) {
  const summaryEl = document.getElementById("edu-summary");
  const bulletsEl = document.getElementById("edu-bullets");
  const defsEl = document.getElementById("edu-definitions");
  const examplesEl = document.getElementById("edu-examples");
  const quizEl = document.getElementById("edu-quiz");

  if (!data || typeof data !== "object") {
    setText(summaryEl, "(No content returned.)");
    bulletsEl.innerHTML = "";
    defsEl.innerHTML = "";
    examplesEl.innerHTML = "";
    quizEl.innerHTML = "";
    return;
  }

  setText(summaryEl, data.summary || "(no summary)");

  bulletsEl.innerHTML = "";
  if (Array.isArray(data.bullets)) {
    data.bullets.forEach((b) => {
      const li = document.createElement("li");
      li.textContent = b;
      bulletsEl.appendChild(li);
    });
  }

  defsEl.innerHTML = "";
  if (Array.isArray(data.definitions)) {
    data.definitions.forEach((d) => {
      const li = document.createElement("li");
      li.textContent = d;
      defsEl.appendChild(li);
    });
  }

  examplesEl.innerHTML = "";
  if (data.example) {
    const div = document.createElement("div");
    div.className = "blockquote";
    div.textContent = data.example;
    examplesEl.appendChild(div);
  } else if (Array.isArray(data.examples)) {
    data.examples.forEach((ex) => {
      const div = document.createElement("div");
      div.className = "blockquote";
      div.textContent = ex;
      examplesEl.appendChild(div);
    });
  }

  quizEl.innerHTML = "";
  const quizKey = data.quick_quiz ? "quick_quiz" : "quiz";
  if (Array.isArray(data[quizKey])) {
    data[quizKey].forEach((qa, idx) => {
      const div = document.createElement("div");
      div.className = "quiz-item";
      const q = document.createElement("div");
      q.className = "quiz-q";
      q.textContent = "Q" + (idx + 1) + ": " + (qa.q || "");
      const a = document.createElement("div");
      a.className = "quiz-a";
      a.textContent = "A" + (idx + 1) + ": " + (qa.a || "");
      div.appendChild(q);
      div.appendChild(a);
      quizEl.appendChild(div);
    });
  }
}

// Small helper to download a string as a .txt file
function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

document.addEventListener("DOMContentLoaded", () => {
  // ====== YouTube → Notes ======
  const ytForm = document.getElementById("yt-form");
  const ytUrl = document.getElementById("yt-url");
  const ytSubmit = document.getElementById("yt-submit");
  const ytStatus = document.getElementById("yt-status");
  const transcriptEl = document.getElementById("transcript");

  const beginnerEl = document.getElementById("notes-beginner");
  const intermediateEl = document.getElementById("notes-intermediate");
  const advancedEl = document.getElementById("notes-advanced");

  const tabBeginner = document.getElementById("tab-beginner");
  const tabIntermediate = document.getElementById("tab-intermediate");
  const tabAdvanced = document.getElementById("tab-advanced");

  const downloadTranscriptBtn = document.getElementById("download-transcript");
  const downloadBeginnerBtn = document.getElementById("download-beginner");
  const downloadIntermediateBtn = document.getElementById("download-intermediate");
  const downloadAdvancedBtn = document.getElementById("download-advanced");

  let lastTranscript = "";
  let lastNotesBeginner = null;
  let lastNotesIntermediate = null;
  let lastNotesAdvanced = null;

  function setActiveTab(level) {
    // Hide all panes
    [beginnerEl, intermediateEl, advancedEl].forEach((el) =>
      el.classList.add("hidden")
    );
    [tabBeginner, tabIntermediate, tabAdvanced].forEach((btn) =>
      btn.classList.remove("active")
    );

    if (level === "beginner") {
      beginnerEl.classList.remove("hidden");
      tabBeginner.classList.add("active");
    } else if (level === "intermediate") {
      intermediateEl.classList.remove("hidden");
      tabIntermediate.classList.add("active");
    } else if (level === "advanced") {
      advancedEl.classList.remove("hidden");
      tabAdvanced.classList.add("active");
    }
  }

  tabBeginner.addEventListener("click", () => setActiveTab("beginner"));
  tabIntermediate.addEventListener("click", () => setActiveTab("intermediate"));
  tabAdvanced.addEventListener("click", () => setActiveTab("advanced"));

  ytForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const url = ytUrl.value.trim();
    if (!url) return;

    ytSubmit.disabled = true;
    ytStatus.textContent = "Processing video on backend...";
    ytStatus.className = "status";
    transcriptEl.textContent = "";
    beginnerEl.textContent = "";
    intermediateEl.textContent = "";
    advancedEl.textContent = "";

    try {
      const res = await fetch("/api/process_youtube", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Unknown error");
      }

      ytStatus.textContent = "Done.";
      ytStatus.className = "status ok";

      lastTranscript = data.transcript || "";
      transcriptEl.textContent = lastTranscript || "(no transcript)";

      lastNotesBeginner = data.notes_beginner || null;
      lastNotesIntermediate = data.notes_intermediate || null;
      lastNotesAdvanced = data.notes_advanced || null;

      renderNotes(beginnerEl, lastNotesBeginner, "beginner");
      renderNotes(intermediateEl, lastNotesIntermediate, "intermediate");
      renderNotes(advancedEl, lastNotesAdvanced, "advanced");

      setActiveTab("beginner");
    } catch (err) {
      ytStatus.textContent = "Error: " + err.message;
      ytStatus.className = "status err";
      console.error(err);
    } finally {
      ytSubmit.disabled = false;
    }
  });

  // Download buttons
  downloadTranscriptBtn.addEventListener("click", () => {
    if (!lastTranscript) {
      alert("No transcript available yet. Run a video first.");
      return;
    }
    downloadText("transcript.txt", lastTranscript);
  });

  downloadBeginnerBtn.addEventListener("click", () => {
    if (!lastNotesBeginner) {
      alert("No beginner notes yet. Process a video first.");
      return;
    }
    downloadText(
      "notes_beginner.json",
      JSON.stringify(lastNotesBeginner, null, 2)
    );
  });

  downloadIntermediateBtn.addEventListener("click", () => {
    if (!lastNotesIntermediate) {
      alert("No intermediate notes yet. Process a video first.");
      return;
    }
    downloadText(
      "notes_intermediate.json",
      JSON.stringify(lastNotesIntermediate, null, 2)
    );
  });

  downloadAdvancedBtn.addEventListener("click", () => {
    if (!lastNotesAdvanced) {
      alert("No advanced notes yet. Process a video first.");
      return;
    }
    downloadText(
      "notes_advanced.json",
      JSON.stringify(lastNotesAdvanced, null, 2)
    );
  });

  // ====== Educational content ======
  const eduForm = document.getElementById("edu-form");
  const eduTopic = document.getElementById("edu-topic");
  const eduLevel = document.getElementById("edu-level");
  const eduSubmit = document.getElementById("edu-submit");
  const eduStatus = document.getElementById("edu-status");
  const downloadEduBtn = document.getElementById("download-edu");

  let lastEduContent = null;

  eduForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const topic = eduTopic.value.trim();
    const level = eduLevel.value;

    if (!topic) return;

    eduSubmit.disabled = true;
    eduStatus.textContent = "Asking Gemini for one-page content...";
    eduStatus.className = "status";

    try {
      const res = await fetch("/api/educational_content", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic, level })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Unknown error");
      }
      eduStatus.textContent = "Done.";
      eduStatus.className = "status ok";
      lastEduContent = data;
      renderEducationalContent(data);
    } catch (err) {
      eduStatus.textContent = "Error: " + err.message;
      eduStatus.className = "status err";
      console.error(err);
    } finally {
      eduSubmit.disabled = false;
    }
  });

  downloadEduBtn.addEventListener("click", () => {
    if (!lastEduContent) {
      alert("No educational content yet. Generate it first.");
      return;
    }
    downloadText(
      "educational_content.json",
      JSON.stringify(lastEduContent, null, 2)
    );
  });
});
