from __future__ import annotations

import shutil
import hashlib
import hmac
import logging
import re
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.schemas import JobCreated, JobHistoryItem, JobHistoryResponse, JobStatus, LogItem, LogResponse
from app.services.batch import SUPPORTED_AUDIO, extract_batch, save_upload, validate_batch
from app.services.errors import BatchValidationError
from app.services.jobs import JobManager
from app.services.auth import AuthStore


app = FastAPI(title="AutoAce Voice Assessment", version="1.0.0")
templates = Jinja2Templates(directory=str(settings.project_root / "app" / "templates"))
templates.env.globals["static_version"] = settings.static_version
app.mount("/static", StaticFiles(directory=str(settings.project_root / "app" / "static")), name="static")
job_manager = JobManager(settings)
auth_store = AuthStore(settings.job_db)
logger = logging.getLogger(__name__)


def _log_warning(message: str, *, job_id: str, artifact: str) -> None:
    logger.warning(message, extra={"job_id": job_id, "artifact": artifact})
    try:
        job_manager.store.add_log("warning", __name__, message, job_id=job_id, artifact=artifact)
    except Exception:
        logger.exception("Could not persist application log")


def require_api_key(request: Request) -> None:
    if not settings.require_api_key:
        return
    if not settings.api_key:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    if _current_user(request):
        return
    provided = request.headers.get("x-api-key", "")
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        provided = authorization[7:]
    session = request.cookies.get("autoace_session", "")
    if session and hmac.compare_digest(session, _session_token()):
        return
    if not hmac.compare_digest(provided, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API credentials")


def _session_token() -> str:
    return hashlib.sha256(f"autoace-session:{settings.api_key}".encode()).hexdigest()


def _current_user(request: Request) -> dict | None:
    return auth_store.get_session_user(request.cookies.get("autoace_session", ""))


def _password_error(password: str) -> str | None:
    if len(password) < 12:
        return "Password must be at least 12 characters"
    return None


def _valid_username(username: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z0-9_.-]{3,32}", username))


@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    if settings.require_api_key and not (_current_user(request) or hmac.compare_digest(request.cookies.get("autoace_session", ""), _session_token())):
        return RedirectResponse("/login", status_code=303)
    response = templates.TemplateResponse(request=request, name="index.html", context={"max_upload_mb": settings.max_upload_bytes // (1024 * 1024), "authenticated": True})
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if _current_user(request):
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request=request, name="login.html", context={"authenticated": False})


@app.post("/login")
async def login(identifier: str = Form(...), password: str = Form(...)):
    user = auth_store.authenticate(identifier, password)
    if not user:
        return RedirectResponse("/login?error=The+email%2C+username%2C+or+password+was+not+accepted", status_code=303)
    token = auth_store.create_session(user["id"])
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        "autoace_session",
        token,
        httponly=True,
        secure=settings.environment == "production",
        samesite="lax",
        max_age=7 * 24 * 60 * 60,
    )
    return response


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    if _current_user(request):
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request=request, name="signup.html", context={"authenticated": False, "allow_signup": settings.allow_signup})


@app.post("/signup")
async def signup(request: Request, email: str = Form(...), username: str = Form(...), password: str = Form(...), confirm_password: str = Form(...)):
    if not settings.allow_signup:
        raise HTTPException(status_code=404, detail="Sign up is not available")
    email = email.strip().lower()
    username = username.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return RedirectResponse("/signup?error=Enter+a+valid+email+address", status_code=303)
    if not _valid_username(username):
        return RedirectResponse("/signup?error=Username+must+be+3-32+characters+using+letters%2C+numbers%2C+periods%2C+hyphens%2C+or+underscores", status_code=303)
    if error := _password_error(password):
        return RedirectResponse(f"/signup?error={error.replace(' ', '+')}", status_code=303)
    if password != confirm_password:
        return RedirectResponse("/signup?error=Passwords+do+not+match", status_code=303)
    try:
        user_id = auth_store.create_user(email, username, password)
    except Exception as exc:
        if "UNIQUE" in str(exc).upper():
            return RedirectResponse("/signup?error=That+email+or+username+is+already+registered", status_code=303)
        raise
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("autoace_session", auth_store.create_session(user_id), httponly=True, secure=settings.environment == "production", samesite="lax", max_age=7 * 24 * 60 * 60)
    return response


@app.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_page(request: Request):
    return templates.TemplateResponse(request=request, name="forgot_password.html", context={"authenticated": False})


@app.post("/forgot-password", response_class=HTMLResponse)
async def forgot_password(request: Request, email: str = Form(...)):
    email = email.strip().lower()
    token = None
    with auth_store._connect() as connection:
        row = connection.execute("SELECT id FROM users WHERE email = ? AND is_active = 1", (email,)).fetchone()
    if row:
        token = auth_store.create_reset_token(row[0])
        logger.info("Password recovery requested for account")
    context = {"authenticated": False, "submitted": True}
    if token and settings.environment != "production":
        context["dev_reset_link"] = f"/reset-password?token={token}"
    return templates.TemplateResponse(request=request, name="forgot_password.html", context=context)


@app.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page(request: Request, token: str = ""):
    return templates.TemplateResponse(request=request, name="reset_password.html", context={"authenticated": False, "token": token})


@app.post("/reset-password")
async def reset_password(token: str = Form(...), password: str = Form(...), confirm_password: str = Form(...)):
    if error := _password_error(password):
        return RedirectResponse(f"/reset-password?token={token}&error={error.replace(' ', '+')}", status_code=303)
    if password != confirm_password:
        return RedirectResponse(f"/reset-password?token={token}&error=Passwords+do+not+match", status_code=303)
    if not auth_store.reset_password(token, password):
        return RedirectResponse("/forgot-password?error=That+recovery+link+is+invalid+or+expired", status_code=303)
    return RedirectResponse("/login?message=Password+updated%2C+please+sign+in", status_code=303)


@app.post("/logout")
async def logout(request: Request):
    token = request.cookies.get("autoace_session", "")
    if token:
        auth_store.delete_session(token)
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("autoace_session")
    return response


@app.get("/health/live")
async def liveness():
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness():
    if not settings.upload_dir.is_dir():
        raise HTTPException(status_code=503, detail="Upload storage is unavailable")
    return {"status": "ready", "engine": settings.inference_engine}


@app.post("/api/jobs", response_model=JobCreated, dependencies=[Depends(require_api_key)])
async def create_job(batch_file: UploadFile = File(...)):
    filename = batch_file.filename or ""
    if not filename.lower().endswith(".zip"):
        await batch_file.close()
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")

    job_id = __import__("uuid").uuid4().hex
    job_dir = settings.upload_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    archive_path = job_dir / "upload.zip"
    try:
        save_upload(batch_file, archive_path, settings)
        extract_batch(archive_path, job_dir, settings)
        archive_path.unlink(missing_ok=True)
        job_manager.submit(job_dir, job_id=job_id, job_type="zip_batch", source_name=filename)
        return JobCreated(job_id=job_id, status="queued")
    except BatchValidationError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        _log_warning(f"Upload rejected: {exc}", job_id=job_id, artifact="upload")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        _log_warning(f"Upload failed: {exc}", job_id=job_id, artifact="upload")
        raise HTTPException(status_code=500, detail="Unable to create processing job") from exc
    finally:
        await batch_file.close()


@app.post("/upload", response_class=HTMLResponse, dependencies=[Depends(require_api_key)])
async def dashboard_upload(request: Request, batch_file: UploadFile = File(...)):
    created = await create_job(batch_file)
    return HTMLResponse(
        f"""<div id='job-status' class='rounded border border-indigo-200 bg-indigo-50 p-4'>
Job <code>{created.job_id}</code> queued. Processing...</div>
<script>
(async function pollJob() {{
  const id = {created.job_id!r};
  const status = document.getElementById('job-status');
  const escapeHtml = value => String(value).replace(/[&<>'\"]/g, character =>
    ({{'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','\"':'&quot;'}}[character]));
  const timer = setInterval(async () => {{
    const response = await fetch('/api/jobs/' + id);
    if (!response.ok) return;
    const job = await response.json();
    status.textContent = 'Job ' + id + ': ' + job.status;
    if (job.status === 'completed' || job.status === 'completed_with_errors') {{
      clearInterval(timer);
      const resultResponse = await fetch('/api/jobs/' + id + '/results');
      const results = await resultResponse.json();
      status.innerHTML = '<pre class="mt-3 overflow-auto text-xs">' +
        escapeHtml(JSON.stringify(results, null, 2)) + '</pre>' +
        '<a class="text-indigo-700 underline" href="/api/jobs/' + id + '/download?format=csv">Download CSV</a>';
    }} else if (job.status === 'failed') {{
      clearInterval(timer);
      status.textContent = 'Job failed: ' + (job.error || 'unknown error');
    }}
  }}, 1000);
}})();
</script>"""
    )


@app.post("/api/jobs/audio", response_model=JobCreated, dependencies=[Depends(require_api_key)])
async def create_audio_job(audio_file: UploadFile = File(...)):
    """Create the normal job lifecycle for one validated audio file."""
    filename = Path((audio_file.filename or "").replace("\\", "/")).name
    if not filename or Path(filename).suffix.lower() not in SUPPORTED_AUDIO:
        await audio_file.close()
        raise HTTPException(status_code=400, detail="Unsupported audio type")
    job_id = __import__("uuid").uuid4().hex
    job_dir = settings.upload_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    try:
        save_upload(audio_file, job_dir / filename, settings)
        validate_batch(job_dir, settings)
        job_manager.submit(job_dir, job_id=job_id, job_type="single_audio", source_name=filename)
        return JobCreated(job_id=job_id, status="queued")
    except BatchValidationError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        _log_warning(f"Audio upload rejected: {exc}", job_id=job_id, artifact="upload")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        _log_warning(f"Audio upload failed: {exc}", job_id=job_id, artifact="upload")
        raise HTTPException(status_code=500, detail="Unable to create processing job") from exc
    finally:
        await audio_file.close()


def _history_item(job: dict) -> JobHistoryItem:
    result_path = Path(job["result_path"]) if job.get("result_path") else None
    error_path = Path(job["error_path"]) if job.get("error_path") else None
    results = []
    errors = []
    if result_path and result_path.is_file():
        import json
        results = json.loads(result_path.read_text(encoding="utf-8"))
    if error_path and error_path.is_file():
        import json
        errors = json.loads(error_path.read_text(encoding="utf-8"))
    confidence = [float(row.get("confidence", 0)) for row in results]
    return JobHistoryItem(
        id=job["id"], status=job["status"], created_at=job["created_at"], updated_at=job["updated_at"],
        job_type=job.get("job_type") or "unknown", source_name=job.get("source_name"),
        total_files=len(results) + len(errors), successful_files=len(results), failed_files=len(errors),
        average_confidence=sum(confidence) / len(confidence) if confidence else None,
        result_available=bool(result_path and result_path.is_file()), error=job.get("error"),
    )


@app.get("/api/jobs/history", response_model=JobHistoryResponse, dependencies=[Depends(require_api_key)])
async def list_history():
    items = [_history_item(job) for job in job_manager.store.list()]
    return JobHistoryResponse(items=items, total=len(items))


@app.get("/api/jobs/{job_id}", response_model=JobStatus, dependencies=[Depends(require_api_key)])
async def get_job(job_id: str):
    job = job_manager.store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**{key: job[key] for key in ("id", "status", "created_at", "updated_at", "error")})


@app.get("/api/jobs/{job_id}/results", dependencies=[Depends(require_api_key)])
async def get_results(job_id: str):
    job = job_manager.store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in {"completed", "completed_with_errors"}:
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}")
    result_path = Path(job["result_path"])
    if not result_path.is_file():
        _log_warning("Result artifact missing", job_id=job_id, artifact="results")
        raise HTTPException(status_code=404, detail="Results are no longer available for this evaluation")
    return FileResponse(result_path, media_type="application/json", filename="results.json")


@app.get("/api/jobs/{job_id}/errors", dependencies=[Depends(require_api_key)])
async def get_errors(job_id: str):
    job = job_manager.store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in {"completed", "completed_with_errors"}:
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}")
    path = Path(job["error_path"])
    if not path.is_file():
        return []
    import json
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/download", dependencies=[Depends(require_api_key)])
async def download_results(job_id: str, format: str = "csv"):
    if format not in {"csv", "json"}:
        raise HTTPException(status_code=400, detail="format must be csv or json")
    job = job_manager.store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in {"completed", "completed_with_errors"}:
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}")
    path = Path(job["result_path"]).with_name(f"results.{format}")
    if not path.is_file():
        _log_warning("Download artifact missing", job_id=job_id, artifact=format)
        raise HTTPException(status_code=404, detail="Result file not found")
    media_type = "text/csv" if format == "csv" else "application/json"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/logs", response_model=LogResponse, dependencies=[Depends(require_api_key)])
async def list_logs(page: int = 1, page_size: int = 10, level: str = "all"):
    if page < 1 or page_size not in {10, 25, 50}:
        raise HTTPException(status_code=400, detail="Invalid logs pagination")
    if level not in {"all", "warning", "error", "critical", "info"}:
        raise HTTPException(status_code=400, detail="Invalid log level")
    rows, total = job_manager.store.list_logs(page, page_size, None if level == "all" else level)
    return LogResponse(items=[LogItem(**row) for row in rows], page=page, page_size=page_size, total=total, page_count=max(1, (total + page_size - 1) // page_size))
