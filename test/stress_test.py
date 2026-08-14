"""Standalone stress runner for the AutoAce voice assessment service.

Spawns uvicorn in an isolated storage directory and drives the live HTTP API
with synthetic audio, checking for crashes, data corruption, temp leaks, and
graceful failure under concurrency. See 1786208841464-stress-test-plan.md.

Usage:
    source autoace-audio-trial/venv/bin/activate
    AUTOACE_REQUIRE_API_KEY=false python test/stress_test.py
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_URL = "http://127.0.0.1:8137"
TIMEOUT_SECONDS = 300
VALID_FIELDS = {
    "emotional_tone", "emotional_intensity", "background_noise_present",
    "background_noise_type", "background_noise_severity", "audio_quality",
    "speaker_overlap_present", "long_silence_present", "confidence", "filename",
}
ALLOWED_TONES = {"neutral", "satisfied", "frustrated", "upset", "distressed"}
ALLOWED_INTENSITY = {"low", "medium", "high"}
ALLOWED_QUALITY = {"clear", "slightly_impaired", "severely_impaired"}


def make_wav_bytes(seconds: float, noise: float = 0.0, tone_hz: float = 220.0) -> bytes:
    """Synthesize a mono 16 kHz int16 WAV: a sine tone plus optional white noise."""
    sample_rate = 16000
    t = np.linspace(0.0, seconds, int(seconds * sample_rate), endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * tone_hz * t)
    if noise:
        signal = signal + noise * np.random.uniform(-1.0, 1.0, t.shape)
    signal = np.clip(signal, -1.0, 1.0)
    output = io.BytesIO()
    sf.write(output, signal.astype(np.float32), sample_rate, format="WAV")
    return output.getvalue()


def make_zip(files: list[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files:
            archive.writestr(name, content)
    return output.getvalue()


def validate_result(row: dict) -> list[str]:
    problems = []
    if set(row) != VALID_FIELDS:
        problems.append(f"fields={sorted(set(row) ^ VALID_FIELDS)}")
    if row.get("emotional_tone") not in ALLOWED_TONES:
        problems.append(f"tone={row.get('emotional_tone')}")
    if row.get("emotional_intensity") not in ALLOWED_INTENSITY:
        problems.append(f"intensity={row.get('emotional_intensity')}")
    if row.get("audio_quality") not in ALLOWED_QUALITY:
        problems.append(f"quality={row.get('audio_quality')}")
    if not (isinstance(row.get("confidence"), (int, float)) and 0.0 <= row["confidence"] <= 1.0):
        problems.append(f"confidence={row.get('confidence')}")
    if not isinstance(row.get("background_noise_present"), bool):
        problems.append("noise_present not bool")
    if not isinstance(row.get("speaker_overlap_present"), bool):
        problems.append("overlap not bool")
    if not isinstance(row.get("long_silence_present"), bool):
        problems.append("silence not bool")
    if row.get("background_noise_severity") not in {"none", "low", "medium", "high"}:
        problems.append(f"severity={row.get('background_noise_severity')}")
    return problems


class StressReport:
    def __init__(self):
        self.status_counts: dict[int, int] = {}
        self.upload_times: list[float] = []
        self.ttc_times: list[float] = []
        self.errors: list[str] = []
        self.schema_problems = 0
        self.jobs_created = 0
        self.jobs_terminal = 0

    def record_status(self, code: int):
        self.status_counts[code] = self.status_counts.get(code, 0) + 1

    def percentile(self, values: list[float], p: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round(p / 100.0 * (len(ordered) - 1))))
        return ordered[index]

    def summary(self) -> str:
        total = sum(self.status_counts.values())
        five_xx = sum(count for code, count in self.status_counts.items() if code >= 500)
        p50_u, p95_u, p99_u = self.percentile(self.upload_times, 50), self.percentile(self.upload_times, 95), self.percentile(self.upload_times, 99)
        p50_t, p95_t, p99_t = self.percentile(self.ttc_times, 50), self.percentile(self.ttc_times, 95), self.percentile(self.ttc_times, 99)
        return (
            f"requests: {total} | 2xx/4xx/5xx: {self.status_counts.get(200, 0)}/"
            f"{self.status_counts.get(400, 0)}/{five_xx}\n"
            f"upload latency p50/p95/p99: {self._fmt(p50_u)}/{self._fmt(p95_u)}/{self._fmt(p99_u)} s\n"
            f"time-to-complete p50/p95/p99: {self._fmt(p50_t)}/{self._fmt(p95_t)}/{self._fmt(p99_t)} s\n"
            f"jobs created: {self.jobs_created} | reached terminal state: {self.jobs_terminal}\n"
            f"schema problems: {self.schema_problems} | errors: {len(self.errors)}"
        )

    @staticmethod
    def _fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"


async def upload_and_wait(
    report: StressReport,
    payload: bytes,
    name: str = "batch.zip",
) -> dict | None:
    """Upload a ZIP with its own connection, poll until terminal, validate schema.

    A fresh client per request avoids httpx's pooled-connection race under
    concurrent uploads and matches how real browsers/clients open connections.
    """
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as client:
        started = time.perf_counter()
        response = await client.post(
            "/api/jobs",
            files={"batch_file": (name, payload, "application/zip")},
            timeout=60,
        )
        upload_latency = time.perf_counter() - started
        report.record_status(response.status_code)
        report.upload_times.append(upload_latency)
        if response.status_code != 200:
            report.errors.append(f"upload {name}: HTTP {response.status_code} {response.text[:200]}")
            return None
        job_id = response.json()["job_id"]
        report.jobs_created += 1

        deadline = time.time() + TIMEOUT_SECONDS
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                status_response = await client.get(f"/api/jobs/{job_id}", timeout=30)
            except httpx.HTTPError as exc:
                last_error = exc
                await asyncio.sleep(0.2)
                continue
            if status_response.status_code != 200:
                report.errors.append(f"poll {job_id}: HTTP {status_response.status_code}")
                break
            status = status_response.json()["status"]
            if status in {"completed", "completed_with_errors", "failed"}:
                report.ttc_times.append(time.perf_counter() - started)
                report.jobs_terminal += 1
                if status == "failed":
                    report.errors.append(f"job {job_id} failed: {status_response.json().get('error')}")
                    return {"job_id": job_id, "status": status, "results": []}
                results_response = await client.get(f"/api/jobs/{job_id}/results", timeout=30)
                results = results_response.json() if results_response.status_code == 200 else []
                for row in results:
                    problems = validate_result(row)
                    if problems:
                        report.schema_problems += 1
                        report.errors.append(f"schema {job_id}: {problems}")
                return {"job_id": job_id, "status": status, "results": results}
            await asyncio.sleep(0.15)
        if last_error:
            report.errors.append(f"poll {job_id} raised {type(last_error).__name__}: {last_error}")
        report.errors.append(f"job {job_id} stuck: no terminal state within {TIMEOUT_SECONDS}s")
        return None


async def scenario_warmup(report: StressReport) -> None:
    payload = make_zip([("warmup.wav", make_wav_bytes(5.0, noise=0.01))])
    outcome = await upload_and_wait(report, payload, "warmup.zip")
    assert outcome and outcome["status"] == "completed" and len(outcome["results"]) == 1


async def scenario_concurrency_ramp(report: StressReport) -> None:
    for level in (5, 10, 20, 50):
        tasks = [
            upload_and_wait(
                report,
                make_zip([(f"c{level}_{index}.wav", make_wav_bytes(10.0, noise=0.05))]),
                f"ramp{level}_{index}.zip",
            )
            for index in range(level)
        ]
        await asyncio.gather(*tasks)
        await asyncio.sleep(3)
        print(f"  ramp {level}: done")


async def scenario_sustained(report: StressReport) -> None:
    window_end = time.time() + 40
    pending: list[asyncio.Task] = []
    index = 0
    while time.time() < window_end:
        batch = [
            asyncio.create_task(
                upload_and_wait(
                    report,
                    make_zip([(f"s{index}_{n}.wav", make_wav_bytes(10.0, noise=0.04))]),
                    f"sustained_{index}_{n}.zip",
                )
            )
            for n in range(20)
        ]
        pending.extend(batch)
        index += 1
        await asyncio.gather(*batch)
    print(f"  sustained: {len(pending)} uploads submitted")


async def scenario_spike(report: StressReport) -> None:
    started = time.perf_counter()
    payloads = [
        make_zip([(f"spike_{index}.wav", make_wav_bytes(5.0, noise=0.03))])
        for index in range(100)
    ]
    tasks = [
        upload_and_wait(report, payloads[index], f"spike_{index}.zip")
        for index in range(100)
    ]
    await asyncio.gather(*tasks)
    print(f"  spike: 100 uploads in {time.perf_counter() - started:.2f}s")


async def scenario_health_during_spike(client: httpx.AsyncClient, report: StressReport) -> None:
    problems = 0
    for _ in range(20):
        try:
            response = await client.get("/health/live", timeout=10)
            if response.status_code != 200:
                problems += 1
        except httpx.HTTPError:
            problems += 1
        await asyncio.sleep(0.15)
    if problems:
        report.errors.append(f"health endpoint failed {problems}/20 times during load")
    else:
        print("  health during load: 20/20 ok")


async def scenario_drain(client: httpx.AsyncClient, report: StressReport) -> None:
    """Wait until the queue fully drains so shutdown-time non-terminal jobs mean a real leak."""
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            response = await client.get("/api/jobs/history", timeout=30)
            items = response.json().get("items", []) if response.status_code == 200 else []
        except httpx.HTTPError:
            await asyncio.sleep(1)
            continue
        pending = [item for item in items if item["status"] not in {"completed", "completed_with_errors", "failed"}]
        if not pending:
            print(f"  drain: all {len(items)} jobs terminal")
            return
        await asyncio.sleep(2)
    report.errors.append(
        f"queue did not drain: {len(pending)}/{len(items)} jobs still non-terminal after 180s "
        f"({sorted({item['status'] for item in pending})})"
    )


async def scenario_edge_cases(client: httpx.AsyncClient, report: StressReport) -> None:
    too_many = make_zip([(f"f{index}.wav", make_wav_bytes(1.0)) for index in range(501)])
    response = await client.post("/api/jobs", files={"batch_file": ("many.zip", too_many, "application/zip")}, timeout=60)
    report.record_status(response.status_code)
    if response.status_code != 400:
        report.errors.append(f"max_files edge: expected 400, got {response.status_code}")
    else:
        print("  max_files edge: 400 as expected")

    big_wav = make_wav_bytes(10.0)
    big_payload = make_zip([(f"big{index}.wav", big_wav) for index in range(200)])
    response = await client.post("/api/jobs", files={"batch_file": ("big.zip", big_payload, "application/zip")}, timeout=60)
    report.record_status(response.status_code)
    if response.status_code != 400:
        report.errors.append(f"size edge: expected 400, got {response.status_code}")
    else:
        print("  size edge: 400 as expected")

    bad_zip = b"this is not a zip archive at all" * 100
    response = await client.post("/api/jobs", files={"batch_file": ("bad.zip", bad_zip, "application/zip")}, timeout=60)
    report.record_status(response.status_code)
    if response.status_code != 400:
        report.errors.append(f"bad zip edge: expected 400, got {response.status_code}")
    else:
        print("  bad zip edge: 400 as expected")


def check_sqlite(storage: Path) -> list[str]:
    import sqlite3

    problems = []
    database = storage / "jobs.sqlite3"
    if not database.is_file():
        return ["sqlite database missing"]
    connection = sqlite3.connect(database)
    try:
        result = connection.execute("PRAGMA integrity_check").fetchone()
        if result[0] != "ok":
            problems.append(f"sqlite integrity: {result[0]}")
        stuck = connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'validating', 'processing')"
        ).fetchone()[0]
        if stuck:
            problems.append(f"sqlite: {stuck} jobs left non-terminal")
        duplicates = connection.execute(
            "SELECT id, COUNT(*) FROM jobs GROUP BY id HAVING COUNT(*) > 1"
        ).fetchall()
        if duplicates:
            problems.append(f"sqlite: duplicate job rows {duplicates}")
    finally:
        connection.close()
    return problems


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="autoace-stress-"))
    storage = workdir / "storage"
    storage.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "AUTOACE_UPLOAD_DIR": str(storage),
            "AUTOACE_JOB_DB": str(storage / "jobs.sqlite3"),
            "AUTOACE_REQUIRE_API_KEY": "false",
            "AUTOACE_INFERENCE_ENGINE": "baseline",
            "AUTOACE_WORKER_COUNT": "2",
            "AUTOACE_MAX_FILES": "500",
            "AUTOACE_MAX_UNCOMPRESSED_BYTES": str(50 * 1024 * 1024),
        }
    )
    server_log = workdir / "server.log"
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8137"],
        cwd=str(PROJECT_ROOT),
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=server_log.open("w"),
    )
    report = StressReport()
    server_crashed = False
    try:
        with httpx.Client(base_url=BASE_URL, timeout=10) as ready_client:
            for _ in range(120):
                if server.poll() is not None:
                    server_crashed = True
                    raise RuntimeError(f"server exited early with {server.returncode}")
                try:
                    if ready_client.get("/health/ready").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.25)
            else:
                raise RuntimeError("server did not become ready")

        async def run():
            async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as client:
                print("scenario: warm-up")
                await scenario_warmup(report)
                print("scenario: concurrency ramp 5/10/20/50")
                await scenario_concurrency_ramp(report)
                print("scenario: sustained load (40s)")
                await scenario_sustained(report)
                print("scenario: spike 100 concurrent uploads")
                await asyncio.gather(
                    scenario_spike(report),
                    scenario_health_during_spike(client, report),
                )
                print("scenario: resource edge cases")
                await scenario_edge_cases(client, report)
                print("scenario: queue drain")
                await scenario_drain(client, report)

        asyncio.run(run())
    except Exception:
        if server.poll() is not None:
            server_crashed = True
        print("\n=== SERVER LOG (last 40 lines) ===")
        lines = server_log.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines[-40:]))
        print(f"\n(artifacts preserved at {workdir})")
        raise
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    if server_crashed:
        print("\n=== SERVER LOG (last 40 lines) ===")
        lines = server_log.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines[-40:]))

    sqlite_problems = check_sqlite(storage)
    job_dirs = len([path for path in storage.iterdir() if path.is_dir()])
    total_status = sum(report.status_counts.values())
    five_xx = sum(count for code, count in report.status_counts.items() if code >= 500)
    leaked = job_dirs - report.jobs_created
    failures = list(report.errors) + sqlite_problems
    if five_xx / total_status > 0.05:
        failures.append(f"5xx rate {five_xx}/{total_status} exceeds 5%")
    if leaked > 2:
        failures.append(f"temp leak: {leaked} job dirs beyond jobs created")
    if report.schema_problems:
        failures.append(f"{report.schema_problems} schema validation problems")

    print("\n=== STRESS REPORT ===")
    print(report.summary())
    print(f"storage: {storage}")
    print(f"sqlite: {'ok' if not sqlite_problems else sqlite_problems}")
    print(f"temp dirs: {job_dirs} | jobs created: {report.jobs_created}")
    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        shutil.rmtree(workdir, ignore_errors=True)
        return 1
    shutil.rmtree(workdir, ignore_errors=True)
    print("\nPASS: no crashes, no schema violations, no leaks, no 5xx spikes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
