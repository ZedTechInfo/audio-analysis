# AutoAce AI

AutoAce AI is a private, browser-based audio evaluation workspace for reviewing
voice emotion, background noise, call quality, speaker overlap, and long
silence. It accepts individual audio files or ZIP batches, processes audio
locally, validates every prediction against a strict schema, and keeps a
searchable history for future reference and comparison.

The project was created for the AutoAce voice-tone technical trial. The
implementation favors deterministic local signal processing for predictable
cost and privacy, with an optional local speech-emotion model for comparison.

## Contents

- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Processing Flow](#processing-flow)
- [Prediction Schema](#prediction-schema)
- [Requirements](#requirements)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Using the Dashboard](#using-the-dashboard)
- [Authentication](#authentication)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Storage](#storage)
- [Security and Privacy](#security-and-privacy)
- [Performance and Cost](#performance-and-cost)
- [Known Limitations](#known-limitations)
- [Project Structure](#project-structure)
- [Operational Notes](#operational-notes)

## Capabilities

### Audio evaluation

- ZIP batch upload with root-level audio files and optional `labels.csv`.
- Single-file testing for WAV, MP3, FLAC, OGG, M4A, and AAC.
- Batch validation for unsupported files, malformed archives, path traversal,
  symbolic links, empty audio, duration limits, and size limits.
- Background processing through a bounded worker pool.
- Per-file failures without aborting an otherwise valid batch.
- JSON and CSV result downloads.

### Signal analysis

- Emotional tone:
  - `neutral`
  - `satisfied`
  - `frustrated`
  - `upset`
  - `distressed`
- Emotional intensity:
  - `low`
  - `medium`
  - `high`
- Background noise presence, type, and severity.
- Audio quality classification.
- Conservative speaker-overlap detection.
- Long-silence detection.
- Evidence-based confidence scoring between `0.0` and `1.0`.

### Workspace

- Dashboard view with live processing status.
- Signal Summary positioned above the upload and status cards.
- Client-side Dashboard, History, and Logs navigation without full page reloads.
- Reactive upload polling while moving between views.
- Paginated history with page sizes of 5, 10, 25, or all records.
- Two-run comparison in a blurred, accessible modal.
- Paginated operational logs.
- Friendly SweetAlert2 error dialogs with technical details logged separately.
- Responsive desktop, tablet, and mobile layouts.
- Reduced-motion support.

## Architecture

AutoAce is a server-rendered FastAPI application with a lightweight reactive
frontend.

```text
Browser
  |
  | Jinja HTML + CSS + vanilla JavaScript
  v
FastAPI application
  |
  +-- Job API --------------------> JobManager / ThreadPoolExecutor
  |                                    |
  |                                    +-- Batch validation
  |                                    +-- Audio analysis
  |                                    +-- JSON/CSV artifacts
  |
  +-- Authentication API ----------> SQLite users and sessions
  |
  +-- History API -----------------> SQLite job metadata + result summaries
  |
  +-- Logs API --------------------> SQLite application_logs
  |
  +-- Static local storage --------> temp_uploads/<job_id>/
```

### Backend

- `app/main.py`
  - FastAPI application and routes.
  - Browser authentication and API-key authentication.
  - Upload handling and artifact downloads.
- `app/services/jobs.py`
  - SQLite job store.
  - Background processing manager.
  - Job history and application-log persistence.
- `app/services/auth.py`
  - User accounts.
  - `scrypt` password hashing.
  - Random expiring sessions.
  - Single-use password-reset tokens.
- `app/services/batch.py`
  - Upload size enforcement.
  - ZIP safety checks.
  - Manifest parsing and audio validation.
- `app/services/audio_engine.py`
  - Waveform loading.
  - Voice activity and silence analysis.
  - Noise detection and severity rules.
  - Audio quality and overlap heuristics.
  - Confidence calibration.
- `app/services/classifier.py`
  - Deterministic heuristic emotion classifier.
  - Optional local SER model integration.
- `app/schemas.py`
  - Pydantic request and response models.
  - Strict `AnalysisResult` prediction contract.

### Frontend

- Jinja partials provide reusable view sections.
- `app/static/dashboard.js` manages upload, polling, results, history, logs,
  comparison, and view-specific interactions.
- `app/static/navigation.js` handles client-side view switching and browser
  history without full document reloads.
- `app/static/pagination.js` centralizes page calculations and bounds.
- `app/static/app.css` contains the visual system, responsive layout, modal,
  navigation, and motion styles.
- SweetAlert2 is used for friendly interaction-error dialogs.

## Processing Flow

1. A user signs in or creates a tester account.
2. The Dashboard accepts a ZIP file or a single audio file.
3. The API validates the upload and creates a UUID job.
4. The job is recorded in SQLite with queued status.
5. A background worker validates the batch contents.
6. Each valid audio file is loaded at 16 kHz mono.
7. Acoustic features are calculated for speech, silence, noise, quality, and
   overlap analysis.
8. The baseline classifier predicts emotion and intensity.
9. Confidence is calibrated from evidence and recording quality.
10. Results and errors are written to the job directory.
11. The frontend polls job status and updates the Dashboard reactively.
12. Completed jobs appear in History and can be reopened or compared.

The baseline path does not send audio to an external inference provider.

## Prediction Schema

Every prediction contains exactly these nine prediction fields:

```json
{
  "emotional_tone": "frustrated",
  "emotional_intensity": "medium",
  "background_noise_present": true,
  "background_noise_type": "office chatter",
  "background_noise_severity": "low",
  "audio_quality": "clear",
  "speaker_overlap_present": false,
  "long_silence_present": false,
  "confidence": 0.74
}
```

### Field rules

| Field | Type | Allowed values or range |
| --- | --- | --- |
| `emotional_tone` | Enum | `neutral`, `satisfied`, `frustrated`, `upset`, `distressed` |
| `emotional_intensity` | Enum | `low`, `medium`, `high` |
| `background_noise_present` | Boolean | `true`, `false` |
| `background_noise_type` | String | Open text; empty when no noise is present |
| `background_noise_severity` | Enum | `none`, `low`, `medium`, `high` |
| `audio_quality` | Enum | `clear`, `slightly_impaired`, `severely_impaired` |
| `speaker_overlap_present` | Boolean | `true`, `false` |
| `long_silence_present` | Boolean | `true`, `false` |
| `confidence` | Number | `0.0` through `1.0` |

`AnalysisResult` rejects unknown fields. Batch exports also include `filename`
as metadata so the original audio file can be identified in JSON and CSV
downloads. `filename` is not part of the prediction contract itself.

## Requirements

- Python 3.12 or compatible Python 3 version supported by the pinned packages.
- `pip`.
- `ffmpeg` may be required by `librosa` for some compressed audio formats.
- At least 2 GB RAM for the baseline path.
- More memory is recommended when using the optional hybrid SER model.
- A modern browser with JavaScript enabled.

The repository includes a project virtual environment at:

```text
autoace-audio-trial/venv/
```

For a clean environment, create a new virtual environment instead of relying
on the checked local environment directory.

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create local configuration:

```bash
cp .env.example .env
```

Edit `.env` before running in a shared or production environment. Do not commit
`.env`, API keys, Hugging Face tokens, or generated audio artifacts.

## Configuration

The example configuration is in `.env.example`.

### Application and authentication

| Variable | Default | Purpose |
| --- | --- | --- |
| `AUTOACE_ENVIRONMENT` | `development` | Controls production cookie security and recovery-link behavior |
| `AUTOACE_API_KEY` | empty | API-key credential for existing API clients |
| `AUTOACE_REQUIRE_API_KEY` | environment-dependent | Protects browser/API routes when enabled |
| `AUTOACE_ALLOW_SIGNUP` | `true` | Enables self-service tester signup |

For production, use a long random API key and set:

```env
AUTOACE_ENVIRONMENT=production
AUTOACE_REQUIRE_API_KEY=true
AUTOACE_ALLOW_SIGNUP=false
```

Self-signup is intended for controlled testing. Password recovery currently
shows a local reset link outside production. A production email provider must
be configured before enabling end-user recovery in production.

### Inference

| Variable | Default | Purpose |
| --- | --- | --- |
| `AUTOACE_INFERENCE_ENGINE` | `baseline` | `baseline` or `hybrid` |
| `AUTOACE_SER_MODEL_ID` | `superb/wav2vec2-base-superb-er` | Local SER checkpoint for hybrid mode |
| `AUTOACE_SER_WINDOW_SECONDS` | `8` | SER inference window |
| `AUTOACE_SER_HOP_SECONDS` | `4` | SER window hop |
| `AUTOACE_WORKER_COUNT` | `1` | Background worker count |

The deterministic baseline is the recommended default because the supplied
pilot showed stronger tone and intensity results with lower operational
complexity. Hybrid mode remains available for controlled comparison.

### Upload and audio limits

| Variable | Default | Purpose |
| --- | ---: | --- |
| `AUTOACE_MAX_UPLOAD_BYTES` | `524288000` | Maximum uploaded archive/file size |
| `AUTOACE_MAX_UNCOMPRESSED_BYTES` | `524288000` | Maximum extracted ZIP size |
| `AUTOACE_MAX_FILES` | `5000` | Maximum files per batch |
| `AUTOACE_MAX_AUDIO_SECONDS` | `3600` | Maximum duration per audio file |
| `AUTOACE_NOISE_FLOOR_THRESHOLD` | `0.001` | Minimum meaningful noise floor |
| `AUTOACE_NOISE_RATIO_THRESHOLD` | `0.03` | Noise-to-speech ratio threshold |
| `AUTOACE_OVERLAP_FRAME_RATIO` | `0.12` | Conservative overlap evidence threshold |

## Running the Application

### Development server

```bash
source "autoace-audio-trial/venv/bin/activate"
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

http://127.0.0.1:8000/
```

When `AUTOACE_REQUIRE_API_KEY=false`, the Dashboard is available directly. To
test account authentication locally, set `AUTOACE_REQUIRE_API_KEY=true`, enable
signup, restart the server, and create a tester account from `/signup`.

### Production-style server

```bash
source "autoace-audio-trial/venv/bin/activate"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Place the application behind HTTPS and a production reverse proxy. The session
cookie is marked `Secure` when `AUTOACE_ENVIRONMENT=production`.

### Health checks

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

Expected responses include:

```json
{"status":"ok"}
```

and:

```json
{"status":"ready","engine":"baseline"}
```

## Using the Dashboard

### Dashboard

The Dashboard provides:

- Current signal summary.
- ZIP or single-audio upload.
- Live job status and progress.
- Structured per-file results.
- CSV and JSON download links.

The frontend changes views through the History API, so moving between Dashboard,
History, and Logs does not interrupt active processing or reload the document.

### History

History provides:

- Search by job ID or source filename.
- Status filtering.
- Sorting by date, confidence, or batch size.
- Page sizes of 5, 10, 25, or all.
- Reopening an existing job.
- Selecting two completed jobs for comparison.

The comparison view opens in a modal with a blurred background. It supports
keyboard closing, focus restoration, and responsive table scrolling.

### Logs

Logs display sanitized structured operational events such as missing result
artifacts. Log records include severity, source, timestamp, job ID, and a safe
message. Sensitive cookies, API keys, filesystem paths, and raw tracebacks are
not intended for the UI.

## Authentication

### Browser sessions

Browser accounts use:

- Email or username login.
- Passwords hashed with Python `hashlib.scrypt`.
- Random session tokens stored as hashes in SQLite.
- Seven-day session expiry.
- HTTP-only cookies.
- `SameSite=Lax` cookies.
- Secure cookies in production.

### Signup

Signup requires:

- Email.
- Username.
- Password of at least 12 characters.
- Password confirmation.

Self-signup can be disabled through `AUTOACE_ALLOW_SIGNUP=false`.

### Password recovery

Password reset tokens are:

- Cryptographically random.
- Stored only as hashes.
- Valid for 30 minutes.
- Single-use.
- Invalidated after a successful password reset.
- Associated with session invalidation after reset.

Recovery responses are non-enumerating: the UI does not disclose whether an
email address exists. In non-production environments, a local reset link is
shown for testing. Production email delivery is not bundled yet.

### API clients

Existing API clients can continue to use:

```http
X-API-Key: <configured-api-key>
```

or:

```http
Authorization: Bearer <configured-api-key>
```

Do not expose API keys in browser-facing UI or commit them to the repository.

## API Reference

### Browser and authentication routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | Authenticated application shell |
| `GET` | `/login` | Sign-in page |
| `POST` | `/login` | Create browser session |
| `POST` | `/logout` | Delete browser session |
| `GET` | `/signup` | Tester signup page |
| `POST` | `/signup` | Create tester account |
| `GET` | `/forgot-password` | Recovery page |
| `POST` | `/forgot-password` | Prepare recovery token |
| `GET` | `/reset-password` | New-password page |
| `POST` | `/reset-password` | Consume recovery token |

### Job routes

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/api/jobs` | Create a ZIP batch job |
| `POST` | `/api/jobs/audio` | Create a single-audio job |
| `GET` | `/api/jobs/{job_id}` | Read job status |
| `GET` | `/api/jobs/{job_id}/results` | Download JSON results |
| `GET` | `/api/jobs/{job_id}/errors` | Read per-file errors |
| `GET` | `/api/jobs/{job_id}/download?format=csv` | Download CSV results |
| `GET` | `/api/jobs/{job_id}/download?format=json` | Download JSON results |
| `GET` | `/api/jobs/history` | List historical jobs |
| `GET` | `/api/logs` | Read paginated sanitized logs |

### Health routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health/live` | Process liveness |
| `GET` | `/health/ready` | Storage and engine readiness |

## Evaluation

The evaluation harness compares predictions with the labeled files in
`evaluation_batch/labels.csv`.

Run the supplied sample:

```bash
source "autoace-audio-trial/venv/bin/activate"
python scripts/run_eval.py evaluation_batch \
  --engine baseline \
  --output-dir output/baseline
```

Run the optional hybrid comparison:

```bash
python scripts/run_eval.py evaluation_batch \
  --engine hybrid \
  --output-dir output/hybrid
```

Generated artifacts include:

- `predictions.json`
- `predictions.csv`
- `errors.json`
- `metrics.json`
- `emotional_tone_confusion_matrix.csv`
- `emotional_tone_confusion_matrix.png` when plotting is available

The supplied three-call set is an integration sample, not a statistically
meaningful hidden-set estimate. The current calibrated baseline sample retains
100% exact field matching while producing differentiated confidence values.
See [EVALUATION.md](EVALUATION.md) and
[docs/accuracy-improvement-plan.md](docs/accuracy-improvement-plan.md) for the
evaluation context and limitations.

## Testing

Run all tests from the project root:

```bash
source "autoace-audio-trial/venv/bin/activate"
python -m unittest discover -s test -v
```

The suite covers:

- ZIP path traversal and archive limits.
- Single-audio validation and upload limits.
- Batch processing and downloads.
- Missing-artifact `404` behavior.
- History summaries.
- Pagination wiring and page calculations.
- Account signup, login, logout, and recovery behavior.
- Password and confidence calibration helpers.
- Prediction schema strictness.

Additional frontend checks:

```bash
node --check app/static/dashboard.js
node --check app/static/navigation.js
node --check app/static/pagination.js
python -m compileall -q app scripts test
```

## Storage

By default, generated and persistent data is stored under `temp_uploads/`:

```text
temp_uploads/
  jobs.sqlite3
  <job-id>/
    results.json
    results.csv
    errors.json
    labels.csv                  # when supplied in the batch
```

SQLite tables include:

- `jobs`
- `users`
- `sessions`
- `password_reset_tokens`
- `application_logs`

Result directories are the source of truth for downloadable per-file results.
The history endpoint keeps job metadata in SQLite and checks whether result
artifacts still exist.

Generated audio, SQLite databases, model caches, and local environment files
should not be committed.

## Security and Privacy

- Audio processing is local by default.
- ZIP extraction rejects path traversal and symbolic links.
- Upload and decompressed-size limits are enforced.
- API credentials are accepted through headers, not query parameters.
- Browser sessions use HTTP-only cookies.
- Production cookies are marked Secure.
- Passwords are never stored in plaintext.
- Reset tokens are never stored in plaintext.
- Login and recovery responses avoid account enumeration.
- Missing artifacts return safe error messages rather than filesystem paths.
- Logs are sanitized before they are displayed in the browser.
- Do not enable public signup in an unrestricted production deployment.
- Put production traffic behind HTTPS and a reverse proxy.

## Performance and Cost

The baseline engine uses local CPU inference and signal processing, so there is
no per-minute external model API charge. Actual infrastructure cost depends on
the host, CPU, memory, concurrency, and storage retention policy.

The optional hybrid engine loads a local Hugging Face SER model and generally
requires more memory. It should be enabled only after validation demonstrates
an accuracy benefit on a representative labeled dataset.

Performance should be measured on the target deployment using:

- Wall-clock seconds per audio minute.
- Peak resident memory.
- Batch size and worker count.
- Model warm-up time.
- Failure rate under concurrent uploads.

## Known Limitations

- The supplied labeled sample contains only three calls.
- Confidence values are evidence estimates, not statistically calibrated
  probabilities until a larger labeled calibration set is available.
- Mono audio cannot reliably prove speaker overlap.
- The baseline classifier is heuristic and should be validated against grouped,
  speaker-aware data before production claims are made.
- The generic SER checkpoint is retained as an opt-in experiment, not the
  default production classifier.
- Production password recovery email delivery still needs an approved provider
  and configuration.
- History currently loads the complete job list before applying client-side
  filtering and pagination.
- Local SQLite storage is appropriate for a single application instance but
  needs a shared database or service for horizontally scaled deployment.

## Project Structure

```text
.
├── app/
│   ├── config.py                 # Environment-backed settings
│   ├── main.py                   # FastAPI routes and app setup
│   ├── schemas.py                # Pydantic contracts
│   ├── services/
│   │   ├── auth.py               # Users, sessions, password reset
│   │   ├── audio_engine.py       # Signal analysis and confidence
│   │   ├── batch.py              # Upload and ZIP validation
│   │   ├── classifier.py         # Emotion inference
│   │   └── jobs.py               # Job processing and SQLite storage
│   ├── static/
│   │   ├── app.css               # Visual system and responsive styles
│   │   ├── dashboard.js          # Reactive application behavior
│   │   ├── navigation.js         # Client-side view navigation
│   │   └── pagination.js          # Shared pagination calculations
│   └── templates/
│       ├── base.html             # Shared document shell
│       ├── index.html            # Single application shell
│       └── partials/             # Reusable dashboard views/components
├── docs/
│   ├── accuracy-improvement-plan.md
│   └── voice-tone-trial-guidelines.md
├── evaluation_batch/             # Supplied labeled evaluation sample
├── scripts/run_eval.py           # Evaluation runner
├── test/                         # Unit and contract tests
├── .env.example                  # Configuration template
├── EVALUATION.md                 # Evaluation harness notes
├── README.md                     # Project overview and operations guide
└── requirements.txt              # Pinned Python dependencies
```

## Operational Notes

### Missing result artifacts

If a completed job references a deleted result file, the API returns a
controlled `404`, the event is recorded in `application_logs`, and the frontend
shows a friendly Close-only modal rather than exposing a stack trace.

### Background workers

Jobs run in a `ThreadPoolExecutor`. The default worker count is one to keep
resource usage predictable. Increasing `AUTOACE_WORKER_COUNT` can improve
throughput but increases CPU and memory pressure.

### Retention

No automatic audio/result retention policy is enabled by default. Operators
should define cleanup rules appropriate to the confidentiality requirements of
the production-call data.

### Browser caching

Authenticated Dashboard responses are marked `no-store` to reduce the chance
that a signed-out user can view a cached authenticated shell through browser
navigation.
