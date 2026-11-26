import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, func
from sqlalchemy.orm import Session, joinedload

from app.core.auth import RequestActor, get_request_actor, require_staff
from app.db.session import get_db
from app.models.company import Company
from app.models.job import Currency, Job, JobLevel, JobStatus
from app.services.crawler_service import CrawlerService

router = APIRouter()


class CrawlBatchRequest(BaseModel):
    urls: List[str] = Field(default_factory=list)
    urls_text: str | None = Field(
        default=None,
        description="Danh sach URL dang text, moi dong 1 link hoac tach bang dau phay/cham phay",
    )


class JobCreateRequest(BaseModel):
    title: str = Field(..., min_length=2, max_length=255)
    company_name: str = Field(default="PrepHire Partner", min_length=1, max_length=255)
    location: Optional[str] = Field(default="Việt Nam", max_length=255)
    level: Optional[str] = Field(default="mid")
    employment_type: Optional[str] = Field(default="Full-time", max_length=50)
    salary_min: Optional[int] = Field(default=None, ge=0)
    salary_max: Optional[int] = Field(default=None, ge=0)
    currency: Optional[str] = Field(default="VND")
    description: str = Field(..., min_length=20)
    role_responsibilities: Optional[str] = None
    skills_qualifications: Optional[str] = None
    benefits: Optional[str] = None
    status: str = Field(default="active")


class JobUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=2, max_length=255)
    company_name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    location: Optional[str] = Field(default=None, max_length=255)
    level: Optional[str] = None
    employment_type: Optional[str] = Field(default=None, max_length=50)
    salary_min: Optional[int] = Field(default=None, ge=0)
    salary_max: Optional[int] = Field(default=None, ge=0)
    currency: Optional[str] = None
    description: Optional[str] = Field(default=None, min_length=10)
    role_responsibilities: Optional[str] = None
    skills_qualifications: Optional[str] = None
    benefits: Optional[str] = None
    status: Optional[str] = None


def _parse_enum(enum_cls, value: Optional[str], default=None):
    if value is None or not str(value).strip():
        return default
    raw = str(value).strip()
    for member in enum_cls:
        if member.value.lower() == raw.lower() or member.name.lower() == raw.lower():
            return member
    raise HTTPException(
        status_code=400,
        detail=f"Giá trị không hợp lệ cho {enum_cls.__name__}: {raw}",
    )


def _get_or_create_company(db: Session, company_name: str, location: Optional[str] = None) -> Company:
    clean_name = company_name.strip()
    existing = (
        db.query(Company)
        .filter(func.lower(Company.name) == clean_name.lower())
        .first()
    )
    if existing:
        return existing
    company = Company(name=clean_name, location=location)
    db.add(company)
    db.flush()
    return company


def _serialize_job_summary(job: Job) -> dict:
    return {
        "job_id": job.job_id,
        "title": job.title,
        "company_name": job.company.name if job.company else None,
        "location": job.location,
        "level": job.level.value if job.level else None,
        "employment_type": job.employment_type,
        "salary_min": job.salary_min,
        "salary_max": job.salary_max,
        "currency": job.currency.value if job.currency else None,
        "status": job.status.value if job.status else "active",
        "source_website": job.source_site,
        "source_url": job.source_url,
        "created_at": job.scraped_at,
    }


def _normalize_batch_urls(payload: CrawlBatchRequest) -> list[str]:
    candidates: list[str] = []
    candidates.extend([item.strip() for item in payload.urls if item and item.strip()])

    if payload.urls_text and payload.urls_text.strip():
        text_parts = re.split(r"[\r\n,;]+", payload.urls_text)
        candidates.extend([item.strip() for item in text_parts if item and item.strip()])

    normalized: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)

    return normalized


@router.post("/crawl")
def crawl_job_api(
    url: str,
    _: RequestActor = Depends(require_staff),
    db: Session = Depends(get_db),
):
    try:
        job = CrawlerService.crawl_job(db, url)
        return {"status": "success", "job": job}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/crawl-batch")
def crawl_jobs_batch_api(
    payload: CrawlBatchRequest,
    _: RequestActor = Depends(require_staff),
    db: Session = Depends(get_db),
):
    urls = _normalize_batch_urls(payload)
    if not urls:
        raise HTTPException(status_code=400, detail="Danh sách urls không được rỗng")

    try:
        results = CrawlerService.crawl_jobs(db, urls)
        success_count = len([item for item in results if item.get("status") == "success"])
        return {
            "status": "success",
            "total": len(results),
            "success": success_count,
            "failed": len(results) - success_count,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/")
def list_jobs(
    limit: int = 50,
    status_filter: Optional[str] = None,
    _: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
):
    query = db.query(Job).options(joinedload(Job.company))
    if status_filter:
        parsed_status = _parse_enum(JobStatus, status_filter)
        query = query.filter(Job.status == parsed_status)
    jobs = query.order_by(desc(Job.scraped_at)).limit(max(1, min(limit, 200))).all()
    return {
        "total": len(jobs),
        "jobs": [_serialize_job_summary(job) for job in jobs],
    }


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_job(
    payload: JobCreateRequest,
    _: RequestActor = Depends(require_staff),
    db: Session = Depends(get_db),
):
    company = _get_or_create_company(db, payload.company_name, payload.location)
    now = datetime.now(timezone.utc)
    job = Job(
        company_company_id=company.company_id,
        title=payload.title.strip(),
        level=_parse_enum(JobLevel, payload.level, default=JobLevel.mid),
        employment_type=payload.employment_type,
        location=payload.location,
        salary_min=payload.salary_min,
        salary_max=payload.salary_max,
        currency=_parse_enum(Currency, payload.currency, default=Currency.VND),
        description_raw=payload.description.strip(),
        description_clean=payload.description.strip(),
        role_responsibilities=payload.role_responsibilities,
        skills_qualifications=payload.skills_qualifications,
        benefits=payload.benefits,
        source_url=f"manual://prephire/{uuid.uuid4().hex}",
        source_site="prephire_admin",
        posted_at=now,
        scraped_at=now,
        status=_parse_enum(JobStatus, payload.status, default=JobStatus.active),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return _serialize_job_summary(job)


@router.patch("/{job_id}")
def update_job(
    job_id: int,
    payload: JobUpdateRequest,
    _: RequestActor = Depends(require_staff),
    db: Session = Depends(get_db),
):
    job = db.query(Job).options(joinedload(Job.company)).filter(Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if payload.company_name is not None:
        company = _get_or_create_company(db, payload.company_name, payload.location or job.location)
        job.company_company_id = company.company_id
    if payload.title is not None:
        job.title = payload.title.strip()
    if payload.location is not None:
        job.location = payload.location.strip()
    if payload.level is not None:
        job.level = _parse_enum(JobLevel, payload.level)
    if payload.employment_type is not None:
        job.employment_type = payload.employment_type
    if payload.salary_min is not None:
        job.salary_min = payload.salary_min
    if payload.salary_max is not None:
        job.salary_max = payload.salary_max
    if payload.currency is not None:
        job.currency = _parse_enum(Currency, payload.currency)
    if payload.description is not None:
        job.description_raw = payload.description.strip()
        job.description_clean = payload.description.strip()
    if payload.role_responsibilities is not None:
        job.role_responsibilities = payload.role_responsibilities
    if payload.skills_qualifications is not None:
        job.skills_qualifications = payload.skills_qualifications
    if payload.benefits is not None:
        job.benefits = payload.benefits
    if payload.status is not None:
        job.status = _parse_enum(JobStatus, payload.status)

    db.commit()
    db.refresh(job)
    return _serialize_job_summary(job)


@router.delete("/{job_id}")
def delete_job(
    job_id: int,
    _: RequestActor = Depends(require_staff),
    db: Session = Depends(get_db),
):
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    db.delete(job)
    db.commit()
    return {"status": "deleted", "job_id": job_id}


@router.get("/{job_id}")
def get_job_detail(
    job_id: int,
    _: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
):
    job = db.query(Job).options(joinedload(Job.company)).filter(Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job.job_id,
        "title": job.title,
        "company_name": job.company.name if job.company else None,
        "location": job.location,
        "level": job.level.value if job.level else None,
        "employment_type": job.employment_type,
        "salary_min": job.salary_min,
        "salary_max": job.salary_max,
        "currency": job.currency.value if job.currency else None,
        "status": job.status.value if job.status else "active",
        "job_description": job.description_clean or job.description_raw,
        "role_responsibilities": job.role_responsibilities,
        "skills_qualifications": job.skills_qualifications,
        "benefits": job.benefits,
        "job_requirements": [
            item.skill.name
            for item in job.job_skills
            if item.skill and item.skill.name
        ],
        "source_website": job.source_site,
        "source_url": job.source_url,
        "created_at": job.scraped_at,
    }


@router.get("/{job_id}/skills")
def get_job_skills(
    job_id: int,
    _: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
):
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    items = []
    for item in job.job_skills:
        if not item.skill or not item.skill.name:
            continue
        items.append(
            {
                "skill_id": item.skill.skill_id,
                "skill_name": item.skill.name,
                "importance": item.importance,
                "required_proficiency": (
                    0.8
                    if (item.importance or 0) >= 0.8
                    else 0.65
                    if (item.importance or 0) >= 0.5
                    else 0.5
                ),
                "evidence_snippet": item.evidence_snippet,
            }
        )

    items.sort(
        key=lambda value: (value["importance"] if value["importance"] is not None else 0),
        reverse=True,
    )

    return {
        "job_id": job.job_id,
        "title": job.title,
        "total_skills": len(items),
        "skills": items,
    }
