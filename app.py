import os
import io
import json
import time
import shutil
import subprocess
import tempfile
import concurrent.futures
import http.client
import wikipedia
import requests
import urllib.request
import urllib.parse
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, session, url_for, send_from_directory
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
import yt_dlp
from faster_whisper import WhisperModel
from werkzeug.security import generate_password_hash, check_password_hash


# ===============================
# ENV
# ===============================
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)


# ===============================
# FLASK
# ===============================
app = Flask(__name__)
# Use a stable fallback in development so debug reloads do not invalidate session cookies.
app.secret_key = SECRET_KEY or "dev-secret-change-me"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"


# ===============================
# FILES
# ===============================
USERS_FILE = Path("users.json")
QUIZ_FILE = Path("quiz_scores.json")

if not USERS_FILE.exists():
    USERS_FILE.write_text("{}")

if not QUIZ_FILE.exists():
    QUIZ_FILE.write_text("{}")


# ===============================
# Whisper (Better Accuracy)
# ===============================
if shutil.which("ffmpeg") is None:
    raise RuntimeError("Install ffmpeg!")

whisper_model = WhisperModel(
    os.getenv("WHISPER_MODEL", "tiny"),
    device="cpu",
    compute_type="int8"
)


# ===============================
# SCHEMAS
# ===============================

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    answer: str


class LearningModule(BaseModel):
    title: str
    summary: str
    key_concepts: List[str]
    definitions: List[str]
    examples: List[str]
    learning_tips: List[str]
    quiz: List[QuizQuestion]


class QuizOnly(BaseModel):
    quiz: List[QuizQuestion]


# ===============================
# AUTH HELPERS
# ===============================

def load_users():
    return json.loads(USERS_FILE.read_text())


def save_users(data):
    USERS_FILE.write_text(json.dumps(data, indent=2))


def require_login():
    return "user" in session


# ===============================
# YOUTUBE AUDIO
# ===============================

def _yt_dlp_opts():
    # yt-dlp now requires an explicit JS runtime for full YouTube extraction.
    js_runtimes = {}
    for runtime in ("node", "deno", "bun"):
        path = shutil.which(runtime)
        if path:
            js_runtimes[runtime] = {"path": path}

    if not js_runtimes:
        raise RuntimeError(
            "No supported JavaScript runtime found for yt-dlp. "
            "Install Node.js (recommended) or Deno."
        )

    return {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'quiet': True,
        'geo_bypass': True,
        'js_runtimes': js_runtimes,
    }


def get_video_info(url):
    # Try multiple extractor configs to reduce false "video unavailable" failures.
    base = _yt_dlp_opts()
    variants = [
        {},
        {'extractor_args': {'youtube': {'player_client': ['android', 'web', 'ios']}}},
        {'extractor_args': {'youtube': {'player_client': ['web_embedded', 'android']}}},
    ]

    # Optional: use browser cookies if explicitly enabled.
    # Set YTDLP_COOKIES_BROWSER=chrome|edge|firefox in .env
    cookie_browser = (os.getenv("YTDLP_COOKIES_BROWSER") or "").strip().lower()
    if cookie_browser:
        variants.append({
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            'cookiesfrombrowser': (cookie_browser,),
        })

    last_err = None
    for patch in variants:
        opts = dict(base)
        opts.update(patch)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            last_err = e
            continue

    if last_err:
        raise last_err
    raise RuntimeError("Failed to extract YouTube video info")


def _clean_error_text(text: str) -> str:
    # Remove ANSI color/control codes from yt-dlp errors.
    return re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text or "").strip()


def _friendly_ytdlp_error(err: Exception) -> str:
    msg = _clean_error_text(str(err))
    lower = msg.lower()
    if "video is not available" in lower:
        return "This YouTube video is not available (removed/private/region-restricted). Try another URL."
    if "private video" in lower:
        return "This is a private YouTube video and cannot be processed."
    if "sign in to confirm your age" in lower or "age-restricted" in lower:
        return "This video is age-restricted and cannot be processed in this setup."
    if "requested format is not available" in lower:
        return "Could not find a playable audio format for this video."
    return msg or "Failed to fetch YouTube video details."


def _is_valid_youtube_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse((url or "").strip())
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]

        if host in ("youtube.com", "m.youtube.com"):
            if parsed.path == "/watch":
                video_id = urllib.parse.parse_qs(parsed.query).get("v", [""])[0]
                return bool(video_id)
            if parsed.path.startswith("/shorts/"):
                return len(parsed.path.split("/")) >= 3 and bool(parsed.path.split("/")[2])
            return False

        if host == "youtu.be":
            return bool(parsed.path.strip("/"))

        return False
    except Exception:
        return False


def get_audio_stream(url, info=None):
    if info is None:
        info = get_video_info(url)

    if not info or "url" not in info:
        raise RuntimeError("Unable to resolve a direct audio stream")

    headers = dict(info.get("http_headers") or {})
    stream_url = info["url"]

    # Resume-capable download for unstable media CDN connections.
    max_attempts = 6
    chunk_size = 256 * 1024
    data = bytearray()
    expected_total = None

    for attempt in range(max_attempts):
        req_headers = dict(headers)
        if data:
            req_headers["Range"] = f"bytes={len(data)}-"

        try:
            with requests.get(stream_url, headers=req_headers, stream=True, timeout=(10, 60)) as resp:
                # If resume is requested and server ignores range, restart from scratch once.
                if data and resp.status_code == 200:
                    data = bytearray()
                    expected_total = None
                resp.raise_for_status()

                if resp.status_code == 206:
                    content_range = resp.headers.get("Content-Range", "")
                    if "/" in content_range:
                        try:
                            expected_total = int(content_range.rsplit("/", 1)[1])
                        except ValueError:
                            pass
                elif resp.status_code == 200 and expected_total is None:
                    length = resp.headers.get("Content-Length")
                    if length and length.isdigit():
                        expected_total = int(length)

                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        data.extend(chunk)

            if not data:
                raise RuntimeError("Resolved audio stream was empty")
            if expected_total is None or len(data) >= expected_total:
                return io.BytesIO(bytes(data))

        except (requests.RequestException, http.client.IncompleteRead):
            pass

        if attempt < max_attempts - 1:
            time.sleep(1.0 + attempt * 0.5)

    raise RuntimeError(
        f"Failed to download full audio stream after retries "
        f"(received {len(data)} bytes{f' of {expected_total}' if expected_total else ''})."
    )


def _choose_caption_track(info):
    sources = [info.get("subtitles") or {}, info.get("automatic_captions") or {}]
    prefer_langs = ["en", "en-us", "en-gb"]
    prefer_exts = ["json3", "srv3", "vtt", "ttml"]

    for source in sources:
        if not source:
            continue
        langs = list(source.keys())
        ordered_langs = []
        for lang in prefer_langs:
            ordered_langs.extend([k for k in langs if k.lower() == lang])
        ordered_langs.extend([k for k in langs if k.lower().startswith("en") and k not in ordered_langs])
        ordered_langs.extend([k for k in langs if k not in ordered_langs])

        for lang in ordered_langs:
            tracks = source.get(lang) or []
            if not tracks:
                continue
            for ext in prefer_exts:
                match = next((t for t in tracks if t.get("ext") == ext and t.get("url")), None)
                if match:
                    return match["url"], ext
            if tracks[0].get("url"):
                return tracks[0]["url"], tracks[0].get("ext", "")
    return None, None


def _clean_caption_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_transcript_from_captions(info):
    url, ext = _choose_caption_track(info)
    if not url:
        return None

    resp = requests.get(url, timeout=20)
    resp.raise_for_status()

    if ext in ("json3", "srv3"):
        data = resp.json()
        chunks = []
        for event in data.get("events", []):
            for seg in event.get("segs", []):
                txt = seg.get("utf8", "")
                if txt:
                    chunks.append(txt)
        text = " ".join(chunks)
        return _clean_caption_text(text) or None

    lines = []
    for raw in resp.text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("WEBVTT", "NOTE", "Kind:", "Language:")):
            continue
        if "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
        lines.append(line)

    text = " ".join(lines)
    return _clean_caption_text(text) or None


def transcribe(audio, max_seconds=None):
    kwargs = {}
    if max_seconds and max_seconds > 0:
        kwargs["clip_timestamps"] = [0, float(max_seconds)]

    segments, _ = whisper_model.transcribe(
        audio,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
        vad_filter=True,
        word_timestamps=False,
        **kwargs,
    )
    return " ".join([s.text for s in segments])


def get_fast_transcript(url):
    info = get_video_info(url)

    caption_text = get_transcript_from_captions(info)
    if caption_text:
        return caption_text, "captions"

    max_seconds = int(os.getenv("FAST_TRANSCRIBE_SECONDS", "120"))
    audio_stream = get_audio_stream(url, info=info)
    return transcribe(audio_stream, max_seconds=max_seconds), "whisper"


# ===============================
# Wikipedia and google books
# ===============================
def get_wikipedia(topic: str) -> str:
    print(topic)
    try:
        wikipedia.set_lang("en")
        summary = wikipedia.summary(topic, sentences=8, auto_suggest=False)
        return f"WIKIPEDIA:\n{summary}"
    except wikipedia.DisambiguationError as e:
        try:
            exact = next((opt for opt in e.options if opt.strip().lower() == topic.strip().lower()), None)
            selected = exact or e.options[0]
            return wikipedia.summary(selected, sentences=6, auto_suggest=False)
        except:
            return ""
    except Exception:
        # Network/API failures from wikipedia should not break the request flow.
        return ""


def get_google_books(topic: str) -> str:
    
    print(topic)
    url = f"https://www.googleapis.com/books/v1/volumes?q={topic}&maxResults=3"

    try:
        res = requests.get(url, timeout=10).json()

        texts = []

        for item in res.get("items", []):
            info = item.get("volumeInfo", {})

            title = info.get("title", "")
            desc = info.get("description", "")

            if desc:
                texts.append(f"{title}: {desc}")

        if not texts:
            return ""

        return "GOOGLE BOOKS:\n" + "\n\n".join(texts)

    except Exception:
        return ""
    
def fetch_reference_sources(topic: str) -> Dict[str, Any]:
    wiki = get_wikipedia(topic)
    books = get_google_books(topic)
    has_wiki = bool((wiki or "").replace("WIKIPEDIA:", "").strip())
    has_books = bool((books or "").replace("GOOGLE BOOKS:", "").strip())
    return {
        "wiki": wiki or "",
        "books": books or "",
        "has_content": has_wiki or has_books,
    }


def build_context(topic: str, wiki: str = "", books: str = "") -> str:

    if not wiki and not books:
        refs = fetch_reference_sources(topic)
        wiki = refs["wiki"]
        books = refs["books"]

    context = f"""
    Teach the exact topic: "{topic}" using VERIFIED academic sources.
    Do not substitute, autocorrect, or reinterpret the topic name.

    {wiki}

    {books}
    """
    return context[:12000]


# ===============================
# GEMINI CORE
# ===============================

def generate_learning(content: str, level: str) -> Dict[str, Any]:
    level_guidance = {
        "Beginner": """
        - Use simple language (short sentences, minimal jargon).
        - Explain terms with everyday analogies.
        - Keep depth introductory and practical.
        - Quiz should test basics only.
        - Return exactly 5 quiz questions.
        """,
        "Intermediate": """
        - Use moderate technical depth with clear definitions.
        - Explain mechanisms and relationships, not just facts.
        - Include cause/effect and common misconceptions.
        - Quiz should mix basics + reasoning.
        - Return exactly 7 quiz questions.
        """,
        "Advanced": """
        - Use technical terminology and detailed mechanisms.
        - Include edge cases, limitations, and nuanced distinctions.
        - Prioritize precision and conceptual rigor.
        - Quiz should include analysis-level questions.
        - Return exactly 10 quiz questions.
        """,
    }.get(level, "")

    system = f"""
        You are an elite AI tutor.

        Teach at {level} level.
        LEVEL-SPECIFIC STYLE:
        {level_guidance}

        STRICT RULES:
        - Use ONLY the provided sources.
        - Do NOT hallucinate facts.
        - If information is missing, say it is unclear.
        - Prefer textbook-style explanations.

        Include quiz questions based ONLY on this content.
        Return STRICT JSON.
    """


    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=content,
        config={
            "system_instruction": system,
            "response_mime_type": "application/json",
            "response_schema": LearningModule
        }
    )

    try:
        return json.loads(response.text)
    except Exception:
        print("Gemini Raw Output:\n", response.text)
        raise RuntimeError("Invalid JSON from Gemini")


def generate_quiz_only(content: str, level: str, count: int) -> List[Dict[str, Any]]:
    prompt = f"""
    Create quiz questions from the provided content for {level} level.
    Return exactly {count} questions.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=content,
        config={
            "system_instruction": prompt,
            "response_mime_type": "application/json",
            "response_schema": QuizOnly
        }
    )
    try:
        parsed = json.loads(response.text)
        return parsed.get("quiz", [])
    except Exception:
        return []


def _normalize_question_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _filter_new_quiz_questions(quiz: List[Dict[str, Any]], excluded: List[str]) -> List[Dict[str, Any]]:
    excluded_set = {_normalize_question_text(q) for q in (excluded or []) if q}
    fresh = []
    seen = set()
    for item in quiz or []:
        question = _normalize_question_text(item.get("question", ""))
        if not question:
            continue
        if question in excluded_set or question in seen:
            continue
        seen.add(question)
        fresh.append(item)
    return fresh


def normalize_quiz_count(learning: Dict[str, Any], level: str, content: str = "") -> Dict[str, Any]:
    target_counts = {
        "Beginner": 5,
        "Intermediate": 7,
        "Advanced": 10,
    }
    target = target_counts.get(level)
    if not target:
        return learning

    quiz = learning.get("quiz", [])
    if not isinstance(quiz, list):
        quiz = []

    if len(quiz) > target:
        learning["quiz"] = quiz[:target]
        return learning

    if len(quiz) < target and content:
        regenerated = generate_quiz_only(content, level, target)
        if isinstance(regenerated, list) and regenerated:
            learning["quiz"] = regenerated[:target]
            return learning

    learning["quiz"] = quiz[:target]
    return learning


def generate_learning_from_topic(topic: str, level: str) -> Dict[str, Any]:
    refs = fetch_reference_sources(topic)

    if refs["has_content"]:
        content = build_context(topic, refs["wiki"], refs["books"])
    else:
        content = f"""
        Teach this exact topic: "{topic}" for a {level} learner.
        Wikipedia and Google Books returned no usable source content.
        Generate the lesson content directly with accurate explanations.
        """

    learning = generate_learning(content, level)
    return normalize_quiz_count(learning, level, content)


def generate_learning_by_level(content: str) -> Dict[str, Dict[str, Any]]:
    levels = [
        ("notes_beginner", "Beginner"),
        ("notes_intermediate", "Intermediate"),
        ("notes_advanced", "Advanced"),
    ]

    results: Dict[str, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        future_map = {
            ex.submit(generate_learning, content, level): key
            for key, level in levels
        }
        for future in concurrent.futures.as_completed(future_map):
            key = future_map[future]
            result = future.result()
            level = next((lvl for k, lvl in levels if k == key), "")
            results[key] = normalize_quiz_count(result, level, content)

    return results


def generate_focused_learning(
    topic: str,
    weak_items: List[Dict[str, str]],
    previous_questions: List[str] = None
) -> Dict[str, Any]:
    weak_lines = []
    for idx, item in enumerate(weak_items, start=1):
        weak_lines.append(
            f"{idx}. Question: {item.get('question', '')}\n"
            f"   Learner answer: {item.get('user_answer', '')}\n"
            f"   Correct answer: {item.get('correct_answer', '')}"
        )

    prev_lines = []
    for idx, q in enumerate(previous_questions or [], start=1):
        prev_lines.append(f"{idx}. {q}")

    prev_block = "\n".join(prev_lines) if prev_lines else "None"

    prompt = f"""
    Topic: {topic}

    The learner scored below or equal to 50% and needs targeted remediation.
    Focus primarily on these weak points:
    {'\n'.join(weak_lines)}

    Previous quiz questions that must NOT be repeated:
    {prev_block}

    Requirements:
    - Explain each weak point in simple, concrete language.
    - Correct misunderstandings directly.
    - Include a short focused quiz to verify improvement.
    - Every new quiz question must be different from the previous quiz list above.
    """
    learning = generate_learning(prompt, "Beginner")
    learning = normalize_quiz_count(learning, "Beginner", prompt)

    filtered = _filter_new_quiz_questions(learning.get("quiz", []), previous_questions or [])
    if len(filtered) < 5:
        refill_prompt = (
            prompt
            + "\nGenerate replacement quiz questions that are all new and non-overlapping "
              "with the previous quiz list."
        )
        regenerated = generate_quiz_only(refill_prompt, "Beginner", 5)
        filtered = _filter_new_quiz_questions(regenerated, previous_questions or [])

    learning["quiz"] = filtered[:5]
    return learning


# ===============================
# EXPLAINER VIDEO
# ===============================
EXPLAINER_ROOT = Path("work/explainers")


def _slugify_topic(topic: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", (topic or "").strip().lower())
    cleaned = re.sub(r"[-\s]+", "_", cleaned)
    return cleaned or "topic"


def generate_explanation(topic: str) -> str:
    prompt = f"""
    Create a deeper explainer script for topic: "{topic}".

    Requirements:
    - Audience: beginner to intermediate learner.
    - Length: about 130 to 180 words.
    - Structure in plain text paragraphs:
      1) Core idea and intuition
      2) Step-by-step working/mechanism
      3) One practical real-world example + one limitation
    - Keep language simple but conceptually deep.
    - Avoid bullet points, markdown, and headings.
    - Write for spoken narration in a video.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
        )
        text = (response.text or "").strip()
        if text:
            return text
    except Exception:
        pass

    return (
        f"{topic} is a practical concept used to solve real-world problems efficiently. "
        f"The core idea is to break a difficult task into clear steps and make decisions using useful signals. "
        f"In most systems, the process starts with inputs, applies a method, and checks output quality before repeating. "
        f"This loop helps improve accuracy and reliability over time. "
        f"For example, teams can use {topic} in routing, prediction, or automation workflows to save time and reduce errors. "
        f"A key limitation is that poor assumptions or noisy data can reduce performance, "
        f"so testing and validation are important before production use."
    )


def generate_audio(text: str, out_dir: Path) -> Path:
    try:
        from gtts import gTTS
    except Exception as e:
        raise RuntimeError("gTTS is not installed. Run: pip install gTTS") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / "audio.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(audio_path)
    wav_path = out_dir / "audio.wav"

    ffmpeg_cmd = shutil.which("ffmpeg")
    if ffmpeg_cmd:
        run = subprocess.run(
            [
                ffmpeg_cmd,
                "-y",
                "-i",
                str(audio_path),
                "-ar",
                "22050",
                "-ac",
                "1",
                str(wav_path),
            ],
            capture_output=True,
            text=True,
        )
        if run.returncode == 0 and wav_path.exists():
            return wav_path

    return audio_path


def _write_manim_scene(topic_slug: str) -> Path:
    scene_code = """
import os
import re
import textwrap
from pathlib import Path
from manim import *

class ExplainerVideo(Scene):
    def _sentences(self, text):
        return [s.strip() for s in re.split(r"(?<=[.!?])\\s+", text) if s.strip()]

    def _section_chunks(self, sentences):
        if not sentences:
            return [
                ("Core Idea", ["No explanation content available."]),
                ("How It Works", []),
                ("Example and Limits", []),
            ]

        labels = ["Core Idea", "How It Works", "Example and Limits"]
        section_size = max(1, (len(sentences) + 2) // 3)
        sections = []
        for idx, label in enumerate(labels):
            start = idx * section_size
            end = (idx + 1) * section_size
            chunk = sentences[start:end]
            if idx == 2 and end < len(sentences):
                chunk = sentences[start:]
            sections.append((label, chunk))
        return sections

    def _visual_for_section(self, idx):
        # Simple topic-agnostic animated visuals for each content section.
        if idx == 0:
            core = Circle(radius=0.45, color=YELLOW, stroke_width=4)
            nodes = VGroup(
                Dot([-1.3, -0.1, 0], color=BLUE_B),
                Dot([1.3, -0.1, 0], color=BLUE_B),
                Dot([0, -1.2, 0], color=BLUE_B),
            )
            links = VGroup(
                Line(core.get_center(), nodes[0].get_center(), color=BLUE_D),
                Line(core.get_center(), nodes[1].get_center(), color=BLUE_D),
                Line(core.get_center(), nodes[2].get_center(), color=BLUE_D),
            )
            return VGroup(links, core, nodes)

        if idx == 1:
            b1 = RoundedRectangle(width=1.5, height=0.65, corner_radius=0.12, color=GREEN_C)
            b2 = RoundedRectangle(width=1.5, height=0.65, corner_radius=0.12, color=GREEN_C)
            b3 = RoundedRectangle(width=1.5, height=0.65, corner_radius=0.12, color=GREEN_C)
            flow = VGroup(b1, b2, b3).arrange(DOWN, buff=0.35)
            arrows = VGroup(
                Arrow(b1.get_bottom(), b2.get_top(), buff=0.04, stroke_width=3, color=GREEN_B),
                Arrow(b2.get_bottom(), b3.get_top(), buff=0.04, stroke_width=3, color=GREEN_B),
            )
            return VGroup(flow, arrows)

        if idx == 2:
            chart = Axes(
                x_range=[0, 4, 1],
                y_range=[0, 5, 1],
                x_length=3.0,
                y_length=2.3,
                axis_config={"color": GREY_B, "include_numbers": False},
                tips=False,
            )
            curve = chart.plot(lambda x: 0.7 + 0.9 * x, color=ORANGE, x_range=[0, 3.6])
            dots = VGroup(
                Dot(chart.c2p(0.5, 1.2), color=ORANGE),
                Dot(chart.c2p(1.7, 2.2), color=ORANGE),
                Dot(chart.c2p(3.0, 3.4), color=ORANGE),
            )
            return VGroup(chart, curve, dots)

        warn = Triangle(color=RED_C, fill_opacity=0.2).scale(0.7)
        warn_text = Text("!", font_size=42, color=RED_B).move_to(warn.get_center())
        shield = RoundedRectangle(width=1.8, height=2.2, corner_radius=0.2, color=TEAL_C)
        shield.next_to(warn, RIGHT, buff=0.35)
        return VGroup(warn, warn_text, shield)

    def construct(self):
        current_project = Path(os.getenv("CURRENT_PROJECT", "."))
        audio_file = current_project / "audio.wav"
        if not audio_file.exists():
            audio_file = current_project / "audio.mp3"
        explanation_file = current_project / "script.txt"

        explanation = explanation_file.read_text(encoding="utf-8") if explanation_file.exists() else "No explanation found."
        sentences = self._sentences(explanation)
        sections = self._section_chunks(sentences)

        bg = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_color="#0b1d4f",
            fill_opacity=1,
            stroke_width=0,
        )
        self.add(bg)

        glow = Circle(radius=2.6, fill_color="#1d4ed8", fill_opacity=0.12, stroke_width=0).shift(UP * 0.9 + LEFT * 2.5)
        self.add(glow)

        topic_name = current_project.name.replace("_", " ")
        title = Text(topic_name, font_size=50, color=YELLOW).to_edge(UP).shift(DOWN * 0.2)
        subtitle = Text("Animated explainer", font_size=23, color=GREY_A).next_to(title, DOWN, buff=0.12)
        divider = Line(LEFT * 5.3, RIGHT * 5.3, color=BLUE_D, stroke_opacity=0.45).next_to(subtitle, DOWN, buff=0.2)

        timeline = NumberLine(
            x_range=[0, 3, 1],
            length=6.2,
            include_ticks=False,
            include_numbers=False,
            color=GREY_B,
        ).next_to(divider, DOWN, buff=0.28)
        progress_dot = Dot(timeline.number_to_point(0), radius=0.08, color=YELLOW)

        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP), run_time=0.8)
        self.play(Create(divider), FadeIn(timeline), FadeIn(progress_dot), run_time=0.8)

        if audio_file.exists():
            self.add_sound(str(audio_file))

        total_words = max(1, len(explanation.split()))
        est_total_seconds = max(20.0, min(65.0, total_words * 0.33))
        per_section_wait = max(2.8, est_total_seconds / max(1, len(sections)))

        card = RoundedRectangle(
            width=10.9,
            height=4.8,
            corner_radius=0.22,
            fill_color="#0f2a5f",
            fill_opacity=0.55,
            stroke_color=BLUE_C,
            stroke_opacity=0.45,
            stroke_width=2,
        ).next_to(timeline, DOWN, buff=0.32)
        self.play(FadeIn(card, shift=UP), run_time=0.9)

        active_group = None
        for idx, (label, chunk) in enumerate(sections):
            sec_title = Text(label, font_size=31, color=TEAL_A).move_to(card.get_top() + DOWN * 0.45)
            visual = self._visual_for_section(idx).scale(0.85).move_to(card.get_left() + RIGHT * 2.0 + DOWN * 0.6)

            chunk_text = " ".join(chunk) if chunk else "No additional details provided."
            wrapped = textwrap.wrap(chunk_text, width=54)
            wrapped = wrapped[:5]
            lines = VGroup(*[
                Text(f"- {line}", font_size=24, color=WHITE)
                for line in wrapped
            ]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            lines.next_to(visual, RIGHT, buff=0.65).shift(UP * 0.1)

            section_group = VGroup(sec_title, visual, lines)

            target_x = min(idx + 1, 3)
            target_point = timeline.number_to_point(target_x)

            if active_group is None:
                self.play(FadeIn(section_group, shift=UP), run_time=1.2)
            else:
                self.play(ReplacementTransform(active_group, section_group), run_time=0.9)

            self.play(progress_dot.animate.move_to(target_point), run_time=0.35)
            active_group = section_group
            self.wait(per_section_wait)

        recap = Text("Quick recap complete", font_size=28, color=YELLOW_B).next_to(card, DOWN, buff=0.2)
        self.play(FadeIn(recap, shift=UP), run_time=0.6)
        self.wait(0.8)
"""
    temp_scene_dir = Path(tempfile.gettempdir()) / "ai_study_buddy_manim"
    temp_scene_dir.mkdir(parents=True, exist_ok=True)
    scene_file = temp_scene_dir / f"vid_{topic_slug}.py"
    scene_file.write_text(scene_code, encoding="utf-8")
    return scene_file


def generate_explainer(topic: str) -> Dict[str, Any]:
    manim_cmd = shutil.which("manim")
    if not manim_cmd:
        raise RuntimeError("manim is not installed or not in PATH.")

    slug = _slugify_topic(topic)
    topic_dir = EXPLAINER_ROOT / slug
    topic_dir.mkdir(parents=True, exist_ok=True)

    explanation = generate_explanation(topic)
    script_file = topic_dir / "script.txt"
    script_file.write_text(explanation, encoding="utf-8")

    generate_audio(explanation, topic_dir)
    scene_file = _write_manim_scene(slug)

    env = dict(os.environ)
    env["CURRENT_PROJECT"] = str(topic_dir.resolve())

    run = subprocess.run(
        [
            manim_cmd,
            "-ql",
            "--fps",
            "18",
            str(scene_file),
            "ExplainerVideo",
            "-o",
            "explainer",
            "--media_dir",
            str(topic_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if run.returncode != 0:
        err = (run.stderr or run.stdout or "").strip()
        raise RuntimeError(f"Manim render failed: {err}")

    videos = sorted(
        topic_dir.glob("videos/**/explainer.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not videos:
        raise RuntimeError("Explainer video file was not created.")

    public_video = topic_dir / "explainer.mp4"
    shutil.copyfile(videos[0], public_video)

    return {
        "topic": topic,
        "slug": slug,
        "script": explanation,
        "video_path": public_video,
    }


# ===============================
# ROOT
# ===============================

@app.route("/")
def root():
    return redirect("/dashboard") if require_login() else redirect("/login")


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


# ===============================
# AUTH
# ===============================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        users = load_users()
        u = request.form["username"]
        p = request.form["password"]

        if u in users and check_password_hash(users[u], p):
            session["user"] = u
            return redirect("/dashboard")

        return "Invalid credentials"

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        users = load_users()
        u = request.form["username"]
        p = request.form["password"]

        if u in users:
            return "User already exists"

        users[u] = generate_password_hash(p)
        save_users(users)

        session["user"] = u
        return redirect("/dashboard")

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ===============================
# DASHBOARD
# ===============================

@app.route("/dashboard")
def dashboard():

    if not require_login():
        return redirect("/login")

    scores = json.loads(QUIZ_FILE.read_text())
    user_scores = scores.get(session["user"], [])

    return render_template("dashboard.html",
                       quiz_scores=user_scores,
                       user=session["user"])



# ===============================
# PAGES
# ===============================
@app.route("/quiz_history")
def quiz_history():

    if not require_login():
        return redirect("/login")

    scores = json.loads(QUIZ_FILE.read_text())
    user_scores = sorted(
    scores.get(session["user"], []),
    key=lambda x: x["date"],
    reverse=True
)


    return render_template(
        "quiz_history.html",
        quiz_scores=user_scores,
        user=session["user"]
    )

@app.route("/topic")
def topic_page():

    if not require_login():
        return redirect("/login")

    return render_template("topic.html", user=session["user"])


@app.route("/youtube")
def youtube_page():

    if not require_login():
        return redirect("/login")

    return render_template("youtube.html", user=session["user"])


@app.route("/explainer")
def explainer_page():

    if not require_login():
        return redirect("/login")

    return render_template("explainer.html", user=session["user"])


@app.route("/quiz")
def quiz_page():

    if not require_login():
        return redirect("/login")

    quiz = session.get("current_quiz")

    if not quiz:
        return redirect("/dashboard")

    return render_template("quiz.html", quiz=quiz)


# ===============================
# APIs
# ===============================

@app.route("/api/topic", methods=["POST"])
def topic_api():

    topic = request.json["topic"]
    level = request.json["level"]

    try:
        learning = generate_learning_from_topic(topic, level)
    except Exception as e:
        return jsonify({"error": f"Failed to generate notes: {str(e)}"}), 500


    session["current_quiz"] = learning["quiz"]
    session["current_topic"] = topic

    return jsonify(learning)


@app.route("/api/youtube", methods=["POST"])
def youtube_api():
    try:
        print(request.json)

        url = (request.json.get("url") or "").strip()
        if not _is_valid_youtube_url(url):
            err = "Invalid YouTube URL. Use a full youtube.com/watch?v=... or youtu.be/... link."
            print(f"[youtube_api] 400: {err} | url={url!r}")
            return jsonify({
                "error": "Invalid YouTube URL. Use a full youtube.com/watch?v=... or youtu.be/... link."
            }), 400
        transcript, transcript_source = get_fast_transcript(url)

        learning_by_level = generate_learning_by_level(transcript)
        advanced_quiz = learning_by_level["notes_advanced"].get("quiz", [])

        session["current_quiz"] = advanced_quiz
        session["current_topic"] = "YouTube Lesson"

        return jsonify({
            "transcript": transcript,
            "transcript_source": transcript_source,
            **learning_by_level,
        })
    except yt_dlp.utils.DownloadError as e:
        err = _friendly_ytdlp_error(e)
        print(f"[youtube_api] 400: {err} | url={url!r}")
        return jsonify({"error": err}), 400
    except Exception as e:
        print(f"[youtube_api] 500: {e}")
        return jsonify({"error": f"Failed to process YouTube URL: {str(e)}"}), 500


@app.route("/api/youtube_transcript", methods=["POST"])
def youtube_transcript_api():
    try:
        url = (request.json.get("url") or "").strip()
        if not _is_valid_youtube_url(url):
            err = "Invalid YouTube URL. Use a full youtube.com/watch?v=... or youtu.be/... link."
            print(f"[youtube_transcript_api] 400: {err} | url={url!r}")
            return jsonify({
                "error": "Invalid YouTube URL. Use a full youtube.com/watch?v=... or youtu.be/... link."
            }), 400
        t0 = time.time()
        transcript, transcript_source = get_fast_transcript(url)
        return jsonify({
            "transcript": transcript,
            "transcript_source": transcript_source,
            "elapsed_seconds": round(time.time() - t0, 2),
        })
    except yt_dlp.utils.DownloadError as e:
        err = _friendly_ytdlp_error(e)
        print(f"[youtube_transcript_api] 400: {err} | url={url!r}")
        return jsonify({"error": err}), 400
    except Exception as e:
        print(f"[youtube_transcript_api] 500: {e}")
        return jsonify({"error": f"Failed to transcribe: {str(e)}"}), 500


@app.route("/api/youtube_notes", methods=["POST"])
def youtube_notes_api():
    try:
        transcript = request.json["transcript"]
        learning_by_level = generate_learning_by_level(transcript)
        advanced_quiz = learning_by_level["notes_advanced"].get("quiz", [])

        session["current_quiz"] = advanced_quiz
        session["current_topic"] = "YouTube Lesson"

        return jsonify(learning_by_level)
    except Exception as e:
        return jsonify({"error": f"Failed to generate notes: {str(e)}"}), 500


@app.route("/api/explainer", methods=["POST"])
def explainer_api():
    try:
        if not require_login():
            return jsonify({"error": "Unauthorized"}), 401

        topic = (request.json.get("topic") or "").strip()
        if not topic:
            return jsonify({"error": "Topic is required."}), 400

        result = generate_explainer(topic)

        return jsonify({
            "topic": result["topic"],
            "script": result["script"],
            "video_url": url_for("explainer_media", slug=result["slug"], filename="explainer.mp4"),
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate explainer video: {str(e)}"}), 500


@app.route("/media/explainer/<slug>/<path:filename>")
def explainer_media(slug, filename):
    if not require_login():
        return redirect("/login")
    base = EXPLAINER_ROOT / _slugify_topic(slug)
    return send_from_directory(base, filename)


# ===============================
# QUIZ
# ===============================

@app.route("/submit_quiz", methods=["POST"])
def submit_quiz():

    answers = request.json["answers"]
    quiz = session.get("current_quiz", [])

    score = sum(
        1 for i, q in enumerate(quiz)
        if i < len(answers) and answers[i] == q["answer"]
    )
    total = len(quiz)
    percentage = (score / total) if total else 0.0

    weak_items = []
    for i, q in enumerate(quiz):
        user_ans = answers[i] if i < len(answers) else ""
        if user_ans != q["answer"]:
            weak_items.append({
                "question": q.get("question", ""),
                "user_answer": user_ans,
                "correct_answer": q.get("answer", ""),
            })

    scores = json.loads(QUIZ_FILE.read_text())
    user = session["user"]

    scores.setdefault(user, []).append({
    "topic": session.get("current_topic"),
    "score": score,
    "total": total,
    "date": time.strftime("%Y-%m-%d %H:%M")
})

    QUIZ_FILE.write_text(json.dumps(scores, indent=2))
    current_topic = session.get("current_topic", "Topic")
    needs_remediation = total > 0 and percentage <= 0.5

    response = {
        "score": score,
        "total": total,
        "percentage": round(percentage * 100, 2),
        "needs_remediation": needs_remediation,
    }

    if needs_remediation and weak_items:
        try:
            previous_questions = [q.get("question", "") for q in quiz]
            focused = generate_focused_learning(current_topic, weak_items, previous_questions)
            response["focused_learning"] = focused
            session["current_quiz"] = focused.get("quiz", [])
            session["current_topic"] = f"Focused Retry: {current_topic}"
        except Exception as e:
            response["remediation_error"] = str(e)
            session.pop("current_quiz", None)
    else:
        session.pop("current_quiz", None)

    return jsonify(response)



# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
