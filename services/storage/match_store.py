"""In-memory stores for API-managed entities that outlive a request.

MatchStore: matches created via POST /matches, their candidate questions
and the answers appended while the match is in progress.
JobStore: diagnostics jobs (202 pattern) with status/result.

Both are UUID-keyed and single-process; they sit behind the same seam as
the rest of the storage layer so a persistent implementation can replace
them without touching the routes.
"""
from __future__ import annotations
import threading
import uuid
from typing import Any, Optional

from services.log_setup import get_logger

log = get_logger('match_store')


def new_id() -> str:
    return uuid.uuid4().hex


class MatchStore:
    """In-progress matches: {match_id: {student_id, status, questions, answers}}."""

    def __init__(self) -> None:
        self._matches: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, student_id: str, questions: list[dict]) -> dict:
        match_id = new_id()
        record = {
            'matchId': match_id,
            'studentId': student_id,
            'status': 'in_progress',
            'questions': questions,
            'answers': [],
        }
        with self._lock:
            self._matches[match_id] = record
        log.info("match created | match_id=%s student_id=%s questions=%d",
                 match_id, student_id, len(questions))
        return record

    def get(self, match_id: str) -> Optional[dict]:
        return self._matches.get(match_id)

    def append_answer(self, match_id: str, answer: dict) -> Optional[dict]:
        with self._lock:
            match = self._matches.get(match_id)
            if match is None:
                return None
            match['answers'].append(answer)
            return match

    def complete(self, match_id: str) -> Optional[dict]:
        with self._lock:
            match = self._matches.get(match_id)
            if match is None:
                return None
            match['status'] = 'completed'
            return match


class JobStore:
    """Diagnostics jobs: {job_id: {jobId, status, result}}."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self) -> dict:
        job_id = new_id()
        job = {'jobId': job_id, 'status': 'processing', 'result': None}
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[dict]:
        return self._jobs.get(job_id)

    def finish(self, job_id: str, result: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job['status'] = 'completed'
                job['result'] = result
        log.info("job completed | job_id=%s result=%s", job_id, result)

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job['status'] = 'failed'
                job['result'] = {'error': error}
        log.error("job failed | job_id=%s error=%s", job_id, error)
