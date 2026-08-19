import gc
import hmac
import html
import json
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import webvtt
import yt_dlp
from fastapi import FastAPI, Header, HTTPException
from faster_whisper import WhisperModel
from pydantic import BaseModel

APP_VERSION = "1.1.0"

TRANSCRIBER_API_KEY = os.getenv("TRANSCRIBER_API_KEY", "").strip()
MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
MODEL_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
MODEL_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
WORK_DIR = Path(os.getenv("WORK_DIR", "/app/tmp"))
JOB_DIR = Path(os.getenv("JOB_DIR", "/app/jobs"))
CPU_THREADS = max(1, int(os.getenv("WHISPER_CPU_THREADS", "2")))
BEAM_SIZE = max(1, int(os.getenv("WHISPER_BEAM_SIZE", "1")))
MAX_VIDEO_MINUTES = max(1, int(os.getenv("MAX_VIDEO_MINUTES", "120")))
JOB_RETENTION_HOURS = max(1, int(os.getenv("JOB_RETENTION_HOURS", "24")))
UNLOAD_MODEL_AFTER_JOB = os.getenv("UNLOAD_MODEL_AFTER_JOB", "true").lower() in {"1", "true", "yes", "on"}

MODEL_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
JOB_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Travel Transcriber",
    version=APP_VERSION,
    description="Local-first YouTube transcript service: manual captions -> auto captions -> faster-whisper fallback.",
)


def verify_api_key(x_api_key: Optional[str]) -> None:
    if not TRANSCRIBER_API_KEY:
        raise HTTPException(status_code=503, detail="TRANSCRIBER_API_KEY is not configured on the server.")
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")
    if not hmac.compare_digest(x_api_key, TRANSCRIBER_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key.")


jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()
job_queue: "queue.Queue[str]" = queue.Queue()
_whisper_model: Optional[WhisperModel] = None
_active_job_id: Optional[str] = None


class CreateJobRequest(BaseModel):
    url: str


YOUTUBE_RE = re.compile(r"^https?://(?:www\.)?(?:youtube\.com|youtu\.be)/", re.IGNORECASE)


def now_ts() -> float:
    return time.time()


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_vtt_time(value: str) -> float:
    value = value.replace(",", ".")
    parts = value.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
    except ValueError:
        pass
    return 0.0


def clean_caption_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u200b", " ")
    return re.sub(r"\s+", " ", text).strip()


def save_job(job: Dict[str, Any]) -> None:
    path = JOB_DIR / f"{job['job_id']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def update_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        job = jobs[job_id]
        job.update(updates)
        job["updated_at"] = now_ts()
        snapshot = dict(job)
    save_job(snapshot)


def load_jobs() -> None:
    cutoff = now_ts() - JOB_RETENTION_HOURS * 3600
    for path in JOB_DIR.glob("*.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            if float(job.get("created_at", 0)) < cutoff:
                path.unlink(missing_ok=True)
                continue
            if job.get("status") in {"queued", "processing"}:
                job["status"] = "queued"
                job["stage"] = "requeued_after_restart"
            jobs[job["job_id"]] = job
        except Exception:
            path.unlink(missing_ok=True)

    for job_id, job in list(jobs.items()):
        if job.get("status") == "queued":
            job_queue.put(job_id)


def cleanup_old_jobs() -> None:
    cutoff = now_ts() - JOB_RETENTION_HOURS * 3600
    with jobs_lock:
        old_ids = [jid for jid, job in jobs.items() if float(job.get("created_at", 0)) < cutoff]
        for jid in old_ids:
            jobs.pop(jid, None)
            (JOB_DIR / f"{jid}.json").unlink(missing_ok=True)


def ydl_base_opts() -> Dict[str, Any]:
    return {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
    }


def inspect_video(url: str) -> Dict[str, Any]:
    opts = ydl_base_opts()
    opts.update({"skip_download": True})
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        raise RuntimeError("yt-dlp returned no video metadata.")
    return info


def pick_language(catalog: Dict[str, Any], detected: Optional[str]) -> Optional[str]:
    if not catalog:
        return None
    keys = [key for key, value in catalog.items() if key != "live_chat" and value]
    if not keys:
        return None

    preferred: List[str] = []
    if detected:
        preferred.extend([detected, detected.split("-")[0]])
    preferred.extend([
        "en", "en-US", "en-GB",
        "zh-Hant", "zh-TW", "zh-Hans", "zh-CN", "zh",
        "ja", "ko", "th", "es", "fr", "de", "it", "pt",
    ])

    for wanted in preferred:
        for key in keys:
            if key.lower() == wanted.lower():
                return key

    for wanted in preferred:
        base = wanted.lower().split("-")[0]
        for key in keys:
            if key.lower().split("-")[0] == base:
                return key

    return keys[0]


def download_caption(url: str, info: Dict[str, Any], temp_dir: Path) -> Optional[Dict[str, Any]]:
    detected = info.get("language")
    manual = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}

    provider = None
    language = pick_language(manual, detected)
    if language:
        provider = "youtube_manual_caption"
    else:
        language = pick_language(automatic, detected)
        if language:
            provider = "youtube_auto_caption"

    if not language or not provider:
        return None

    before = set(temp_dir.iterdir())
    opts = ydl_base_opts()
    opts.update({
        "skip_download": True,
        "writesubtitles": provider == "youtube_manual_caption",
        "writeautomaticsub": provider == "youtube_auto_caption",
        "subtitleslangs": [language],
        "subtitlesformat": "vtt/best",
        "outtmpl": str(temp_dir / "%(id)s.%(ext)s"),
    })

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)

    after = set(temp_dir.iterdir())
    new_files = list(after - before)
    vtt_files = [p for p in new_files if p.suffix.lower() == ".vtt"] or list(temp_dir.glob("*.vtt"))
    if not vtt_files:
        return None

    segments: List[Dict[str, Any]] = []
    previous_text = None
    for caption in webvtt.read(str(vtt_files[0])):
        text = clean_caption_text(caption.text)
        if not text or text == previous_text:
            continue
        start = parse_vtt_time(caption.start)
        end = parse_vtt_time(caption.end)
        segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})
        previous_text = text

    if not segments:
        return None

    transcript = " ".join(x["text"] for x in segments)
    timestamped = "\n".join(f"[{fmt_time(x['start'])}] {x['text']}" for x in segments)
    return {
        "provider": provider,
        "language": language,
        "transcript": transcript,
        "timestamped_transcript": timestamped,
        "segments": segments,
    }


def download_audio(url: str, temp_dir: Path) -> Path:
    opts = ydl_base_opts()
    opts.update({
        "format": "bestaudio[abr<=128]/bestaudio/best",
        "outtmpl": str(temp_dir / "%(id)s.%(ext)s"),
    })

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        prepared = Path(ydl.prepare_filename(info))

    if prepared.exists():
        return prepared

    candidates = [
        p for p in temp_dir.iterdir()
        if p.is_file() and p.suffix.lower() not in {".vtt", ".json", ".part"}
    ]
    if not candidates:
        raise RuntimeError("Audio download completed but no audio file was found.")
    return max(candidates, key=lambda p: p.stat().st_size)


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            MODEL_NAME,
            device=MODEL_DEVICE,
            compute_type=MODEL_COMPUTE_TYPE,
            cpu_threads=CPU_THREADS,
            num_workers=1,
            download_root=str(MODEL_DIR),
        )
    return _whisper_model


def unload_whisper_model() -> None:
    global _whisper_model
    if _whisper_model is not None:
        del _whisper_model
        _whisper_model = None
        gc.collect()


def transcribe_audio(audio_path: Path) -> Dict[str, Any]:
    model = get_whisper_model()
    segments_gen, info = model.transcribe(
        str(audio_path),
        beam_size=BEAM_SIZE,
        vad_filter=True,
        condition_on_previous_text=True,
        word_timestamps=False,
    )
    raw_segments = list(segments_gen)
    segments = [
        {
            "start": round(float(s.start), 3),
            "end": round(float(s.end), 3),
            "text": str(s.text).strip(),
        }
        for s in raw_segments if str(s.text).strip()
    ]
    if not segments:
        raise RuntimeError("Whisper produced an empty transcript.")

    transcript = " ".join(x["text"] for x in segments)
    timestamped = "\n".join(f"[{fmt_time(x['start'])}] {x['text']}" for x in segments)
    return {
        "provider": "local_whisper",
        "language": getattr(info, "language", None) or "unknown",
        "language_probability": round(float(getattr(info, "language_probability", 0) or 0), 4),
        "transcript": transcript,
        "timestamped_transcript": timestamped,
        "segments": segments,
    }


def process_job(job_id: str) -> None:
    global _active_job_id
    with jobs_lock:
        url = jobs[job_id]["url"]

    temp_dir = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=str(WORK_DIR)))
    _active_job_id = job_id

    try:
        update_job(job_id, status="processing", stage="inspect_youtube")
        info = inspect_video(url)

        metadata = {
            "video_id": info.get("id"),
            "title": info.get("title"),
            "channel": info.get("channel") or info.get("uploader"),
            "duration_seconds": info.get("duration"),
            "webpage_url": info.get("webpage_url") or url,
        }
        update_job(job_id, stage="checking_captions", metadata=metadata)

        caption_result = download_caption(url, info, temp_dir)
        if caption_result:
            update_job(job_id, status="completed", stage="completed", **caption_result)
            return

        duration = float(info.get("duration") or 0)
        if duration and duration > MAX_VIDEO_MINUTES * 60:
            raise RuntimeError(
                f"No YouTube captions were available and the video is {duration / 60:.1f} minutes long. "
                f"Local Whisper fallback is limited to {MAX_VIDEO_MINUTES} minutes."
            )

        update_job(job_id, stage="downloading_audio")
        audio_path = download_audio(url, temp_dir)
        update_job(job_id, stage="local_whisper")
        whisper_result = transcribe_audio(audio_path)
        update_job(job_id, status="completed", stage="completed", **whisper_result)

    except Exception as exc:
        update_job(job_id, status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if UNLOAD_MODEL_AFTER_JOB:
            unload_whisper_model()
        _active_job_id = None
        cleanup_old_jobs()


def worker_loop() -> None:
    while True:
        job_id = job_queue.get()
        try:
            process_job(job_id)
        finally:
            job_queue.task_done()


@app.on_event("startup")
def startup() -> None:
    load_jobs()
    threading.Thread(target=worker_loop, daemon=True, name="transcription-worker").start()


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "travel-transcriber",
        "version": APP_VERSION,
        "status": "ok",
        "authentication": "X-API-Key required for /jobs endpoints",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "travel-transcriber",
        "version": APP_VERSION,
        "queue_size": job_queue.qsize(),
        "active_job_id": _active_job_id,
        "whisper_model": MODEL_NAME,
        "whisper_model_loaded": _whisper_model is not None,
        "max_whisper_concurrency": 1,
        "api_key_configured": bool(TRANSCRIBER_API_KEY),
    }


@app.post("/jobs", status_code=202)
def create_job(
    payload: CreateJobRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Dict[str, Any]:
    verify_api_key(x_api_key)

    url = payload.url.strip()
    if not YOUTUBE_RE.match(url):
        raise HTTPException(status_code=400, detail="Only YouTube/youtu.be URLs are accepted.")

    job_id = uuid.uuid4().hex
    created = now_ts()
    job = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "url": url,
        "created_at": created,
        "updated_at": created,
    }

    with jobs_lock:
        jobs[job_id] = job
    save_job(job)
    job_queue.put(job_id)

    return {"job_id": job_id, "status": "queued", "poll_url": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Dict[str, Any]:
    verify_api_key(x_api_key)

    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        path = JOB_DIR / f"{job_id}.json"
        if path.exists():
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                job = None

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    return job
