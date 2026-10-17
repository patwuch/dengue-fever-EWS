"""Local API for triggering GPU retraining in the dengue-ews Docker image.

Runs on the host (not in a container) so it can shell out to `docker run`.
Meant for a single local user — bind it to 127.0.0.1 unless you've thought
through exposing it further; see README.md.
"""

import os
import subprocess
import threading
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="dengue-ews training API")

IMAGE = os.environ.get("DENGUE_IMAGE", "dengue-ews")
DATA_DIR = os.environ.get("DENGUE_DATA_DIR", os.path.abspath("data"))
RESULTS_DIR = os.environ.get(
    "DENGUE_RESULTS_DIR", os.path.abspath("machine-learning-module/results")
)


class Job:
    def __init__(self, job_id: str, command: list[str]):
        self.id = job_id
        self.command = command
        self.status = "running"  # running | succeeded | failed
        self.exit_code: int | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.log_lines: list[str] = []
        self.lock = threading.Lock()


jobs: dict[str, Job] = {}
current_job_id: str | None = None
state_lock = threading.Lock()


class RunRequest(BaseModel):
    target: str
    configfile: str
    cores: int = 4
    gpu: int = 1


def _stream_job(job: Job, proc: subprocess.Popen):
    global current_job_id
    for line in proc.stdout:
        with job.lock:
            job.log_lines.append(line.rstrip("\n"))
    proc.wait()
    with job.lock:
        job.exit_code = proc.returncode
        job.status = "succeeded" if proc.returncode == 0 else "failed"
        job.finished_at = time.time()
    with state_lock:
        current_job_id = None


@app.post("/jobs", status_code=202)
def start_job(req: RunRequest):
    global current_job_id
    with state_lock:
        if current_job_id is not None:
            raise HTTPException(
                status_code=409,
                detail=f"job {current_job_id} is already running — only one GPU job at a time",
            )

        command = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{DATA_DIR}:/workspace/data",
            "-v", f"{RESULTS_DIR}:/workspace/machine-learning-module/results",
        ]
        if "WANDB_API_KEY" in os.environ:
            # bare "-e WANDB_API_KEY" (no "=value") makes docker read the value
            # from this process's own env at run time, so the secret never
            # appears in argv, subprocess logs, or the /jobs/{id} response below.
            command += ["-e", "WANDB_API_KEY"]
        command += [
            IMAGE,
            "snakemake", req.target,
            "--configfile", req.configfile,
            "--cores", str(req.cores),
            "--resources", f"gpu={req.gpu}",
        ]

        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id, command)
        jobs[job_id] = job
        current_job_id = job_id
        threading.Thread(target=_stream_job, args=(job, proc), daemon=True).start()

    return {"job_id": job_id, "status": "running"}


@app.get("/jobs")
def list_jobs():
    return [
        {"job_id": j.id, "status": j.status, "started_at": j.started_at}
        for j in jobs.values()
    ]


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    with job.lock:
        return {
            "job_id": job.id,
            "status": job.status,
            "exit_code": job.exit_code,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "command": " ".join(job.command),
        }


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str, tail: int = 200):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    with job.lock:
        return {"job_id": job.id, "lines": job.log_lines[-tail:]}


@app.get("/health")
def health():
    return {"status": "ok", "current_job": current_job_id}
