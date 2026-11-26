from __future__ import annotations

import json
import time
from urllib.parse import urlparse

import requests
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.models.company import Company
from app.models.cv import CV
from app.models.cv_analysis_result import CvAnalysisResult
from app.models.job import Job
from app.services.interview_service import InterviewSessionService
from app.services.pdf_processor import CvIngestService


MAX_CV_BYTES = 5 * 1024 * 1024
OPTIONS_CACHE_TTL_SECONDS = 60.0
_options_cache: dict[int, tuple[float, dict]] = {}
ALLOWED_CV_HOST_SUFFIXES = (
    ".cloudflarestorage.com",
    ".r2.dev",
    ".supabase.co",
)


class StoredInterviewDemoService:
    @staticmethod
    def _allowed_cv_url(url: str) -> bool:
        parsed = urlparse(url or "")
        host = (parsed.hostname or "").lower()
        return parsed.scheme == "https" and any(
            host.endswith(suffix) for suffix in ALLOWED_CV_HOST_SUFFIXES
        )

    @classmethod
    def _download_cv(cls, cv: CV) -> tuple[str, str]:
        if not cls._allowed_cv_url(cv.file_url):
            raise ValueError("Stored CV URL is not on an approved storage host")
        try:
            with requests.get(
                cv.file_url,
                timeout=(5, 20),
                stream=True,
                allow_redirects=False,
            ) as response:
                response.raise_for_status()
                declared_size = int(response.headers.get("content-length") or 0)
                if declared_size > MAX_CV_BYTES:
                    raise ValueError("Stored CV exceeds the 5MB limit")
                chunks: list[bytes] = []
                received = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > MAX_CV_BYTES:
                        raise ValueError("Stored CV exceeds the 5MB limit")
                    chunks.append(chunk)
                file_bytes = b"".join(chunks)
                content_type = response.headers.get("content-type")
        except requests.RequestException as exc:
            raise ValueError(f"Could not download stored CV: {exc}") from exc
        text = CvIngestService.extract_text_from_file(
            cv.file_name,
            content_type,
            file_bytes,
        )
        return text, "stored_file"

    @staticmethod
    def _analysis_fallback(db: Session, cv: CV) -> tuple[str, str]:
        analysis = (
            db.query(CvAnalysisResult)
            .filter(CvAnalysisResult.cv_filename == cv.file_name)
            .order_by(desc(CvAnalysisResult.created_at))
            .first()
        )
        if not analysis:
            raise ValueError(
                "CV file is unavailable and no previous extracted profile was found"
            )
        excerpt = (analysis.cv_text_excerpt or "").strip()
        profile = analysis.extracted_profile_json or {}
        text = (
            f"Stored CV file: {cv.file_name}\n"
            f"CV excerpt:\n{excerpt}\n\n"
            "Previously extracted structured profile:\n"
            f"{json.dumps(profile, ensure_ascii=False)}"
        ).strip()
        if len(text) < 30:
            raise ValueError("Previous CV extraction does not contain enough text")
        return text, "analysis_fallback"

    @classmethod
    def load_cv_text(cls, db: Session, cv: CV) -> tuple[str, str]:
        try:
            return cls._download_cv(cv)
        except (ValueError, OSError):
            return cls._analysis_fallback(db, cv)

    @staticmethod
    def build_job_text(job: Job) -> str:
        sections = [
            f"Job title: {job.title}",
            f"Company: {job.company.name}" if job.company else "",
            f"Level: {job.level.value}" if job.level else "",
            f"Location: {job.location}" if job.location else "",
            job.description_clean or job.description_raw or "",
            job.role_responsibilities or "",
            job.skills_qualifications or "",
            job.benefits or "",
        ]
        unique: list[str] = []
        for section in sections:
            normalized = section.strip()
            if normalized and normalized not in unique:
                unique.append(normalized)
        text = "\n\n".join(unique)
        if len(text) < 30:
            raise ValueError("Selected job does not contain enough description text")
        return text

    @staticmethod
    def list_options(db: Session, limit: int = 100) -> dict:
        normalized_limit = max(1, min(limit, 100))
        now = time.monotonic()
        cached = _options_cache.get(normalized_limit)
        if cached and (now - cached[0]) < OPTIONS_CACHE_TTL_SECONDS:
            return cached[1]

        analyses = {
            str(name).strip().casefold()
            for (name,) in db.query(CvAnalysisResult.cv_filename).distinct().all()
            if name
        }
        cv_rows = (
            db.query(CV.cv_id, CV.file_name, CV.uploaded_at)
            .order_by(desc(CV.uploaded_at))
            .limit(300)
            .all()
        )
        cvs: list[dict] = []
        seen_names: set[str] = set()
        for cv_id, file_name, uploaded_at in cv_rows:
            key = (file_name or "").strip().casefold()
            if not key or key in seen_names:
                continue
            seen_names.add(key)
            cvs.append(
                {
                    "cv_id": cv_id,
                    "file_name": file_name,
                    "uploaded_at": uploaded_at,
                    "fallback_ready": key in analyses,
                }
            )
            if len(cvs) >= normalized_limit:
                break

        job_rows = (
            db.query(
                Job.job_id,
                Job.title,
                Company.name.label("company_name"),
                Job.location,
                Job.level,
            )
            .outerjoin(Company, Job.company_company_id == Company.company_id)
            .order_by(desc(Job.scraped_at), desc(Job.job_id))
            .limit(normalized_limit)
            .all()
        )
        jobs = [
            {
                "job_id": job_id,
                "title": title,
                "company_name": company_name,
                "location": location,
                "level": level.value if level else None,
            }
            for job_id, title, company_name, location, level in job_rows
        ]
        result = {"cvs": cvs, "jobs": jobs}
        _options_cache[normalized_limit] = (now, result)
        return result

    @classmethod
    def create_session(
        cls,
        db: Session,
        *,
        cv_id: int,
        job_id: int,
        language: str,
        max_questions: int,
        allow_external_ai: bool,
        user_id: int | None = None,
    ):
        cv = db.query(CV).filter(CV.cv_id == cv_id).first()
        if not cv:
            raise ValueError("Stored CV not found")
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            raise ValueError("Stored job not found")
        cv_text, _source = cls.load_cv_text(db, cv)
        job_text = cls.build_job_text(job)
        return InterviewSessionService.create(
            db=db,
            cv_text=cv_text,
            jd_text=job_text,
            language=language,
            max_questions=max_questions,
            allow_external_ai=allow_external_ai,
            research_consent=False,
            user_id=user_id if user_id is not None else cv.user_user_id,
            context_source="stored_cv_job",
            source_cv_id=cv.cv_id,
            source_job_id=job.job_id,
        )
