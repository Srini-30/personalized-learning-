document.addEventListener("DOMContentLoaded", () => {
  const ytForm = document.getElementById("yt-form");
  const ytUrl = document.getElementById("yt-url");
  const ytSubmit = document.getElementById("yt-submit");
  const ytStatus = document.getElementById("yt-status");
  const transcriptEl = document.getElementById("transcript") || null;

  const beginnerEl = document.getElementById("notes-beginner");
  const intermediateEl = document.getElementById("notes-intermediate");
  const advancedEl = document.getElementById("notes-advanced");

  const tabBeginner = document.getElementById("tab-beginner");
  const tabIntermediate = document.getElementById("tab-intermediate");
  const tabAdvanced = document.getElementById("tab-advanced");

  const downloadBeginner = document.getElementById("download-beginner");
  const downloadIntermediate = document.getElementById("download-intermediate");
  const downloadAdvanced = document.getElementById("download-advanced");

  const quizBtn = document.getElementById("quizBtn");
  const currentTopic = "YouTube Lesson";

  let currentQuiz = null;
  let notesByLevel = {
    beginner: null,
    intermediate: null,
    advanced: null,
  };

  if (!ytForm || !ytUrl || !ytSubmit || !ytStatus) return;

  ytForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const url = ytUrl.value.trim();
    if (!url) return;

    ytSubmit.disabled = true;
    ytStatus.textContent = "Processing video...";

    if (transcriptEl) transcriptEl.textContent = "";
    beginnerEl.textContent = "";
    intermediateEl.textContent = "";
    advancedEl.textContent = "";
    if (quizBtn) quizBtn.style.display = "none";

    try {
      const transcriptRes = await fetch("/api/youtube_transcript", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      const transcriptData = await transcriptRes.json();
      if (!transcriptRes.ok) throw new Error(transcriptData.error || "Server error");

      if (transcriptEl) transcriptEl.textContent = transcriptData.transcript || "";
      ytStatus.textContent = "Transcript ready. Generating notes...";

      const notesRes = await fetch("/api/youtube_notes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: transcriptData.transcript }),
      });

      const data = await notesRes.json();
      if (!notesRes.ok) throw new Error(data.error || "Server error");
      ytStatus.textContent = "Done";

      notesByLevel = {
        beginner: data.notes_beginner || null,
        intermediate: data.notes_intermediate || null,
        advanced: data.notes_advanced || null,
      };

      renderLearning(beginnerEl, notesByLevel.beginner);
      renderLearning(intermediateEl, notesByLevel.intermediate);
      renderLearning(advancedEl, notesByLevel.advanced);

      currentQuiz = notesByLevel.beginner?.quiz || null;

      if (currentQuiz && quizBtn) quizBtn.style.display = "block";
      showTab("beginner");
    } catch (err) {
      ytStatus.textContent = "Error: " + err.message;
      console.error(err);
    } finally {
      ytSubmit.disabled = false;
    }
  });

  if (quizBtn) {
    quizBtn.addEventListener("click", () => {
      if (!currentQuiz) {
        alert("Quiz not ready yet!");
        return;
      }
      sessionStorage.setItem("quiz", JSON.stringify(currentQuiz));
      sessionStorage.setItem("topic", currentTopic);
      window.location.href = "/quiz";
    });
  }

  if (tabBeginner) tabBeginner.addEventListener("click", () => showTab("beginner"));
  if (tabIntermediate) tabIntermediate.addEventListener("click", () => showTab("intermediate"));
  if (tabAdvanced) tabAdvanced.addEventListener("click", () => showTab("advanced"));

  if (downloadBeginner) {
    downloadBeginner.addEventListener("click", () => downloadNotes("beginner", notesByLevel.beginner));
  }
  if (downloadIntermediate) {
    downloadIntermediate.addEventListener("click", () => downloadNotes("intermediate", notesByLevel.intermediate));
  }
  if (downloadAdvanced) {
    downloadAdvanced.addEventListener("click", () => downloadNotes("advanced", notesByLevel.advanced));
  }

  function showTab(level) {
    const byLevel = {
      beginner: beginnerEl,
      intermediate: intermediateEl,
      advanced: advancedEl,
    };
    const tabs = {
      beginner: tabBeginner,
      intermediate: tabIntermediate,
      advanced: tabAdvanced,
    };

    Object.keys(byLevel).forEach((key) => {
      if (!byLevel[key] || !tabs[key]) return;
      byLevel[key].classList.toggle("hidden", key !== level);
      tabs[key].classList.toggle("active", key === level);
    });

    currentQuiz = notesByLevel[level]?.quiz || null;
    if (quizBtn) quizBtn.style.display = currentQuiz ? "block" : "none";
  }

  function renderLearning(el, data) {
    el.innerHTML = "";
    if (!data) {
      el.textContent = "No notes.";
      return;
    }

    if (data.title) {
      const title = document.createElement("h3");
      title.textContent = data.title;
      el.appendChild(title);
    }

    if (data.summary) {
      const summary = document.createElement("div");
      summary.className = "blockquote";
      summary.textContent = data.summary;
      el.appendChild(summary);
    }

    renderList(el, "Key Concepts", data.key_concepts);
    renderList(el, "Definitions", data.definitions);
    renderList(el, "Examples", data.examples);
    renderList(el, "Learning Tips", data.learning_tips);
  }

  function renderList(parent, heading, items) {
    if (!Array.isArray(items) || items.length === 0) return;

    const h = document.createElement("h4");
    h.textContent = heading;
    parent.appendChild(h);

    const ul = document.createElement("ul");
    items.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      ul.appendChild(li);
    });
    parent.appendChild(ul);
  }

  function downloadNotes(level, notes) {
    if (!notes) {
      alert("No notes available to download yet.");
      return;
    }

    const blob = new Blob([JSON.stringify(notes, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    const safeTopic = currentTopic.toLowerCase().replace(/\s+/g, "-");
    a.href = URL.createObjectURL(blob);
    a.download = `${safeTopic}-${level}-notes.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }
});

function startQuiz() {
  window.location.href = "/quiz";
}
