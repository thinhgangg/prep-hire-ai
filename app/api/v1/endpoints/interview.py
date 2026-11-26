import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.auth import (
    RequestActor,
    assert_session_access,
    get_request_actor,
    require_staff,
)
from app.db.session import engine, get_db
from app.models.candidate_evaluation import CandidateEvaluation
from app.models.expert_annotation import ExpertAnnotation
from app.models.interview_evidence_summary import InterviewEvidenceSummary
from app.models.interview_gaze_report import InterviewGazeReport
from app.models.interview_review import InterviewReview
from app.models.interview_session import InterviewSession
from app.models.interview_turn import InterviewTurn
from app.models.user_data import User
from app.schemas.interview import (
    ContextExtractionMetadata,
    CandidateEvaluationRequest,
    CandidateEvaluationResponse,
    CandidateRubricScores,
    ExpertAnnotationRequest,
    ExpertAnnotationResponse,
    ExpertConsensusResponse,
    InterviewAnswerRequest,
    InterviewContextRequest,
    InterviewContextResponse,
    InterviewEvidenceSummaryResponse,
    InterviewGazeEventPayload,
    InterviewGazeReportRequest,
    InterviewGazeReportResponse,
    InterviewQuestionCandidateResponse,
    InterviewQuestionResponse,
    InterviewReviewRequest,
    InterviewReviewResponse,
    InterviewSessionCreateRequest,
    InterviewSessionResponse,
    InterviewTurnResponse,
    RealtimeInterviewModelStatusResponse,
    StoredInterviewOptionsResponse,
    StoredInterviewSessionCreateRequest,
    SyntheticInterviewSessionCreateRequest,
)
from app.research.rubrics import CANDIDATE_EVALUATION_RUBRIC, QUESTION_QUALITY_RUBRIC
from app.services.evaluation_service import CandidateEvaluationService, ExpertConsensusService
from app.services.interview_evidence_aggregation_service import (
    InterviewEvidenceSummaryService,
)
from app.services.interview_model_service import InterviewModelService
from app.services.interview_service import AdaptiveQuestionEngine, InterviewSessionService
from app.services.pdf_processor import CvIngestService
from app.services.stored_interview_demo_service import StoredInterviewDemoService
from app.services.synthetic_interview_context_service import SyntheticInterviewContextService


router = APIRouter()

_MAX_GAZE_SNAPSHOT_EVENTS = 12
_MAX_SNAPSHOT_DATA_LEN = 350_000


def _ensure_gaze_and_review_tables(db: Session) -> None:
    try:
        bind = db.get_bind()
        if bind is not None:
            InterviewGazeReport.__table__.create(bind=bind, checkfirst=True)
            InterviewReview.__table__.create(bind=bind, checkfirst=True)
    except Exception:
        pass


def _load_reviews_map(db: Session, session_ids: list[str]) -> dict[str, InterviewReview]:
    if not session_ids:
        return {}
    try:
        rows = (
            db.query(InterviewReview)
            .filter(InterviewReview.session_id.in_(session_ids))
            .all()
        )
        return {row.session_id: row for row in rows}
    except SQLAlchemyError:
        db.rollback()
        return {}


def _load_users_map(db: Session, user_ids: list[int | None]) -> dict[int, User]:
    clean_ids = sorted({uid for uid in user_ids if isinstance(uid, int) and uid > 0})
    if not clean_ids:
        return {}
    try:
        rows = db.query(User).filter(User.user_id.in_(clean_ids)).all()
        return {row.user_id: row for row in rows}
    except SQLAlchemyError:
        db.rollback()
        return {}


def _review_response(review, session_id: str) -> InterviewReviewResponse | None:
    if not review:
        return None
    return InterviewReviewResponse(
        review_id=getattr(review, "review_id", None),
        session_id=session_id,
        shortlisted=bool(getattr(review, "shortlisted", False)),
        decision=str(getattr(review, "decision", "pending") or "pending"),
        hr_notes=getattr(review, "hr_notes", None),
        reviewed_by_user_id=getattr(review, "reviewed_by_user_id", None),
        reviewed_by_email=getattr(review, "reviewed_by_email", None),
        updated_at=getattr(review, "updated_at", None),
    )


def _gaze_response(report: InterviewGazeReport) -> InterviewGazeReportResponse:
    events = [
        InterviewGazeEventPayload(**item)
        for item in (report.gaze_events_json or [])
        if isinstance(item, dict) and item.get("timestamp")
    ]
    return InterviewGazeReportResponse(
        report_id=report.report_id,
        session_id=report.session_id,
        total_frames=report.total_frames,
        face_frames=report.face_frames,
        duration_sec=report.duration_sec,
        camera_active=bool(report.camera_active),
        face_presence_pct=report.face_presence_pct,
        gaze_center_pct=report.gaze_center_pct,
        gaze_left_pct=report.gaze_left_pct,
        gaze_right_pct=report.gaze_right_pct,
        gaze_away_pct=report.gaze_away_pct,
        gaze_events=events,
        gaze_timeline=list(report.gaze_timeline_json or []),
        integrity_score=getattr(report, "integrity_score", None),
        integrity_rank=getattr(report, "integrity_rank", None),
        integrity_summary=getattr(report, "integrity_summary_json", None),
        created_at=report.created_at,
        updated_at=report.updated_at,
    )


def _session_response(
    session,
    *,
    include_candidates: bool = True,
    review=None,
    user=None,
) -> InterviewSessionResponse:
    turns = []
    for turn in session.turns:
        raw_candidates = (
            (getattr(turn, "question_candidates_json", None) or [])
            if include_candidates
            else []
        )
        candidate_items = [
            InterviewQuestionCandidateResponse(
                provider_name=item.get("provider"),
                model_name=item.get("model_name"),
                question=str(item.get("question") or ""),
                criterion=item.get("criterion"),
                question_type=item.get("question_type"),
                difficulty=item.get("difficulty"),
                pairwise_utility=item.get("pairwise_utility"),
                selected=bool(item.get("selected", False)),
                generation_latency_ms=int(
                    item.get("generation_latency_ms") or 0
                ),
            )
            for item in raw_candidates
            if item.get("question")
        ]
        turns.append(
            InterviewTurnResponse(
                turn_index=turn.turn_index,
                stage=turn.stage,
                question=turn.question_text,
                answer=turn.answer_text,
                question_source=turn.question_source,
                provider_name=getattr(turn, "provider_name", None),
                model_name=turn.model_name,
                generation_latency_ms=turn.generation_latency_ms,
                question_candidate_count=len(candidate_items) or 1,
                question_candidates=candidate_items,
                selected_question_score=getattr(turn, "selected_question_score", None),
                question_ranker_version=getattr(turn, "question_ranker_version", None),
                question_selection_latency_ms=getattr(
                    turn, "question_selection_latency_ms", 0
                ),
                answer_evidence=getattr(turn, "answer_scores_json", None),
                answer_scorer_version=getattr(turn, "answer_scorer_version", None),
                answer_scoring_latency_ms=getattr(
                    turn, "answer_scoring_latency_ms", 0
                ),
                created_at=turn.created_at,
                answered_at=turn.answered_at,
            )
        )
    active_turn = next(
        (turn for turn in turns if turn.turn_index == session.current_turn_index and turn.answer is None),
        None,
    )
    current_question = None
    if active_turn:
        current_question = InterviewQuestionResponse(
            turn_index=active_turn.turn_index,
            stage=active_turn.stage,
            question=active_turn.question,
        )
    return InterviewSessionResponse(
        session_id=session.session_id,
        user_id=getattr(session, "user_id", None),
        candidate_name=getattr(user, "name", None) if user else None,
        candidate_email=getattr(user, "email", None) if user else None,
        status=session.status,
        language=session.language,
        max_questions=session.max_questions,
        current_stage=session.current_stage,
        allow_external_ai=session.allow_external_ai,
        research_consent=session.research_consent,
        context_source=getattr(session, "context_source", "user_supplied"),
        source_cv_id=getattr(session, "source_cv_id", None),
        source_job_id=getattr(session, "source_job_id", None),
        cv_profile_json=getattr(session, "cv_profile_json", None),
        job_profile_json=getattr(session, "job_profile_json", None),
        current_question=current_question,
        turns=turns,
        review=_review_response(review, session.session_id),
        created_at=session.created_at,
        completed_at=session.completed_at,
    )


def _evaluation_response(evaluation) -> CandidateEvaluationResponse:
    return CandidateEvaluationResponse(
        evaluation_id=evaluation.evaluation_id,
        session_id=evaluation.session_id,
        provider=evaluation.provider,
        model_name=evaluation.model_name,
        prompt_version=evaluation.prompt_version,
        scores=CandidateRubricScores(**evaluation.scores_json),
        evidence=evaluation.evidence_json,
        overall_explanation=evaluation.overall_explanation,
        latency_ms=evaluation.latency_ms,
        created_at=evaluation.created_at,
    )


def _evidence_summary_response(summary) -> InterviewEvidenceSummaryResponse:
    return InterviewEvidenceSummaryResponse(
        summary_id=summary.summary_id,
        session_id=summary.session_id,
        aggregator_version=summary.aggregator_version,
        status=summary.status,
        competencies=summary.competency_scores_json,
        evidence=summary.evidence_json,
        flags=summary.flags_json,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
    )


def _annotation_response(annotation) -> ExpertAnnotationResponse:
    return ExpertAnnotationResponse(
        annotation_id=annotation.annotation_id,
        session_id=annotation.session_id,
        expert_id=annotation.expert_id,
        rubric_version=annotation.rubric_version,
        scores=CandidateRubricScores(**annotation.scores_json),
        notes=annotation.notes,
        created_at=annotation.created_at,
        updated_at=annotation.updated_at,
    )


def _resolve_bound_user_id(actor: RequestActor, requested_user_id: int | None) -> int | None:
    if not actor.is_staff and actor.user_id is not None:
        return actor.user_id
    return requested_user_id if requested_user_id is not None else actor.user_id


@router.get("/rubrics")
def get_research_rubrics() -> dict:
    return {
        "question_quality": QUESTION_QUALITY_RUBRIC,
        "candidate_evaluation": CANDIDATE_EVALUATION_RUBRIC,
    }


@router.get(
    "/realtime-models/status",
    response_model=RealtimeInterviewModelStatusResponse,
)
def get_realtime_model_status() -> RealtimeInterviewModelStatusResponse:
    return RealtimeInterviewModelStatusResponse(**InterviewModelService.status())


@router.post(
    "/sessions/upload",
    response_model=InterviewSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session_from_upload(
    cv_file: UploadFile | None = File(default=None),
    cv_id: int | None = Form(default=None),
    cv_text: str | None = Form(default=None),
    job_id: int | None = Form(default=None),
    jd_file: UploadFile | None = File(default=None),
    jd_text: str | None = Form(default=None),
    job_title: str | None = Form(default=None),
    job_company: str | None = Form(default=None),
    language: str = Form(default="vi"),
    max_questions: int = Form(default=5),
    allow_external_ai: bool = Form(default=True),
    user_id: int | None = Form(default=None),
    actor: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
) -> InterviewSessionResponse:
    bound_uid = _resolve_bound_user_id(actor, user_id)

    final_cv_text = ""
    if cv_file and cv_file.filename:
        cv_bytes = await cv_file.read()
        final_cv_text = CvIngestService.extract_text_from_file(
            cv_file.filename, cv_file.content_type, cv_bytes
        )
    elif cv_text and len(cv_text.strip()) >= 30:
        final_cv_text = cv_text.strip()
    elif cv_id:
        from app.models.cv import CV

        cv_row = db.query(CV).filter(CV.cv_id == cv_id).first()
        if cv_row:
            final_cv_text, _ = StoredInterviewDemoService.load_cv_text(db, cv_row)

    if not final_cv_text or len(final_cv_text) < 30:
        raise HTTPException(
            status_code=400,
            detail="Vui lòng tải lên file CV (PDF/DOCX/TXT) hoặc chọn một CV hợp lệ",
        )

    final_jd_text = ""
    resolved_job_title = (job_title or "").strip() or None
    resolved_job_company = (job_company or "").strip() or None
    if jd_file and jd_file.filename:
        jd_bytes = await jd_file.read()
        final_jd_text = CvIngestService.extract_text_from_file(
            jd_file.filename, jd_file.content_type, jd_bytes
        )
    elif jd_text and len(jd_text.strip()) >= 30:
        final_jd_text = jd_text.strip()
    elif job_id:
        from app.models.job import Job

        job_row = db.query(Job).filter(Job.job_id == job_id).first()
        if job_row:
            final_jd_text = StoredInterviewDemoService.build_job_text(job_row)
            resolved_job_title = job_row.title
            if job_row.company:
                resolved_job_company = job_row.company.name

    if not final_jd_text or len(final_jd_text) < 30:
        raise HTTPException(
            status_code=400,
            detail="Vui lòng chọn vị trí tuyển dụng hoặc nhập/tải lên mô tả công việc (JD)",
        )

    try:
        session = InterviewSessionService.create(
            db=db,
            cv_text=final_cv_text,
            jd_text=final_jd_text,
            language=language,
            max_questions=max_questions,
            allow_external_ai=allow_external_ai,
            research_consent=False,
            user_id=bound_uid,
            context_source="stored_job" if job_id else "user_upload",
            source_cv_id=cv_id,
            source_job_id=job_id,
            job_title=resolved_job_title,
            job_company=resolved_job_company,
        )
    except (ValueError, SQLAlchemyError) as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Lỗi khởi tạo phiên phỏng vấn: {exc}"
        ) from exc

    return _session_response(session)


@router.post(
    "/sessions",
    response_model=InterviewSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_interview_session(
    payload: InterviewSessionCreateRequest,
    actor: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
) -> InterviewSessionResponse:
    bound_uid = _resolve_bound_user_id(actor, payload.user_id)
    try:
        session = InterviewSessionService.create(
            db=db,
            cv_text=payload.cv_text,
            jd_text=payload.jd_text,
            language=payload.language,
            max_questions=payload.max_questions,
            allow_external_ai=payload.allow_external_ai,
            research_consent=payload.research_consent,
            user_id=bound_uid,
        )
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to create interview session: {exc}"
        ) from exc
    return _session_response(session)


@router.post(
    "/sessions/files",
    response_model=InterviewSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_interview_session_from_cv_file(
    cv_file: UploadFile = File(..., description="CV: PDF, DOCX, TXT, or MD"),
    jd_text: str = Form(..., min_length=30),
    language: str = Form(default="vi"),
    max_questions: int = Form(default=8, ge=5, le=15),
    allow_external_ai: bool = Form(default=True),
    research_consent: bool = Form(default=False),
    actor: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
) -> InterviewSessionResponse:
    normalized_language = language.strip().lower()
    if normalized_language not in {"vi", "en"}:
        raise HTTPException(status_code=400, detail="language must be 'vi' or 'en'")

    try:
        cv_bytes = await cv_file.read()
        cv_text = CvIngestService.extract_text_from_file(
            cv_file.filename or "cv",
            cv_file.content_type,
            cv_bytes,
        )
        session = InterviewSessionService.create(
            db=db,
            cv_text=cv_text,
            jd_text=jd_text,
            language=normalized_language,
            max_questions=max_questions,
            allow_external_ai=allow_external_ai,
            research_consent=research_consent,
            user_id=actor.user_id,
        )
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create interview session: {exc}",
        ) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to read CV file: {exc}"
        ) from exc

    return _session_response(session)


@router.post(
    "/synthetic-sessions",
    response_model=InterviewSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_synthetic_interview_session(
    payload: SyntheticInterviewSessionCreateRequest,
    _: RequestActor = Depends(require_staff),
    db: Session = Depends(get_db),
) -> InterviewSessionResponse:
    context = SyntheticInterviewContextService.generate(
        payload.seed,
        payload.role,
        payload.level,
    )
    try:
        session = InterviewSessionService.create(
            db=db,
            cv_text=context.cv_text,
            jd_text=context.jd_text,
            language=payload.language,
            max_questions=payload.max_questions,
            allow_external_ai=payload.allow_external_ai,
            research_consent=False,
            context_source="synthetic",
        )
    except (ValueError, SQLAlchemyError) as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create synthetic interview session: {exc}",
        ) from exc
    return _session_response(session)


@router.get("/sessions", response_model=list[InterviewSessionResponse])
def list_all_sessions(
    limit: int = 100,
    _: RequestActor = Depends(require_staff),
    db: Session = Depends(get_db),
) -> list[InterviewSessionResponse]:
    sessions = InterviewSessionService.list_all(db, limit=limit)
    reviews_map = _load_reviews_map(db, [s.session_id for s in sessions])
    users_map = _load_users_map(db, [getattr(s, "user_id", None) for s in sessions])
    return [
        _session_response(
            s,
            include_candidates=False,
            review=reviews_map.get(s.session_id),
            user=users_map.get(getattr(s, "user_id", None)),
        )
        for s in sessions
    ]


@router.get("/users/{user_id}/sessions", response_model=list[InterviewSessionResponse])
def list_user_sessions(
    user_id: int,
    actor: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
) -> list[InterviewSessionResponse]:
    if not actor.is_staff and actor.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bạn chỉ được xem danh sách phiên phỏng vấn của chính mình.",
        )
    sessions = InterviewSessionService.list_for_user(db, user_id)
    reviews_map = _load_reviews_map(db, [s.session_id for s in sessions])
    users_map = _load_users_map(db, [user_id])
    owner = users_map.get(user_id)
    return [
        _session_response(
            s,
            include_candidates=False,
            review=reviews_map.get(s.session_id),
            user=owner,
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}", response_model=InterviewSessionResponse)
def get_interview_session(
    session_id: str,
    actor: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
) -> InterviewSessionResponse:
    session = InterviewSessionService.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    assert_session_access(session, actor)
    _ensure_gaze_and_review_tables(db)
    reviews_map = _load_reviews_map(db, [session.session_id])
    users_map = _load_users_map(db, [getattr(session, "user_id", None)])
    return _session_response(
        session,
        review=reviews_map.get(session.session_id),
        user=users_map.get(getattr(session, "user_id", None)),
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_200_OK)
def delete_interview_session(
    session_id: str,
    actor: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
) -> dict:
    session = InterviewSessionService.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    assert_session_access(session, actor)

    try:
        for child_model in (
            InterviewEvidenceSummary,
            InterviewGazeReport,
            InterviewReview,
            CandidateEvaluation,
            ExpertAnnotation,
            InterviewTurn,
        ):
            try:
                with db.begin_nested():
                    db.query(child_model).filter(
                        child_model.session_id == session_id
                    ).delete(synchronize_session=False)
            except SQLAlchemyError:
                pass
        db.query(InterviewSession).filter(
            InterviewSession.session_id == session_id
        ).delete(synchronize_session=False)
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to delete interview session: {exc}"
        ) from exc
    return {"status": "deleted", "session_id": session_id}


@router.post("/sessions/{session_id}/answers", response_model=InterviewSessionResponse)
def submit_interview_answer(
    session_id: str,
    payload: InterviewAnswerRequest,
    actor: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
) -> InterviewSessionResponse:
    existing = InterviewSessionService.get(db, session_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Interview session not found")
    assert_session_access(existing, actor)

    try:
        session = InterviewSessionService.submit_answer(db, session_id, payload.answer)
    except ValueError as exc:
        db.rollback()
        status_code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to save interview answer: {exc}"
        ) from exc
    return _session_response(session)


# ============================================================================
# Eye-Gaze Persistence Endpoints
# ============================================================================


@router.put("/sessions/{session_id}/gaze", response_model=InterviewGazeReportResponse)
def upsert_session_gaze_report(
    session_id: str,
    payload: InterviewGazeReportRequest,
    actor: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
) -> InterviewGazeReportResponse:
    session = InterviewSessionService.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    assert_session_access(session, actor)
    _ensure_gaze_and_review_tables(db)

    sanitized_events: list[dict] = []
    snapshot_count = 0
    for ev in payload.gaze_events[:60]:
        ev_dict = ev.model_dump(mode="json")
        snap = ev_dict.get("snapshot")
        if snap:
            if snapshot_count >= _MAX_GAZE_SNAPSHOT_EVENTS or len(snap) > _MAX_SNAPSHOT_DATA_LEN:
                ev_dict["snapshot"] = None
            else:
                snapshot_count += 1
        sanitized_events.append(ev_dict)

    now = datetime.now(timezone.utc)
    try:
        report = (
            db.query(InterviewGazeReport)
            .filter(InterviewGazeReport.session_id == session_id)
            .first()
        )
        if report is None:
            report = InterviewGazeReport(
                session_id=session_id,
                created_at=now,
            )
            db.add(report)

        report.total_frames = payload.total_frames
        report.face_frames = payload.face_frames
        report.duration_sec = payload.duration_sec
        report.camera_active = payload.camera_active
        report.face_presence_pct = payload.face_presence_pct
        report.gaze_center_pct = payload.gaze_center_pct
        report.gaze_left_pct = payload.gaze_left_pct
        report.gaze_right_pct = payload.gaze_right_pct
        report.gaze_away_pct = payload.gaze_away_pct
        report.gaze_events_json = sanitized_events
        report.gaze_timeline_json = payload.gaze_timeline[:300]
        report.integrity_score = payload.integrity_score
        report.integrity_rank = payload.integrity_rank
        report.integrity_summary_json = payload.integrity_summary
        report.updated_at = now
        db.commit()
        db.refresh(report)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to save gaze report: {exc}"
        ) from exc
    return _gaze_response(report)


@router.get("/sessions/{session_id}/gaze", response_model=InterviewGazeReportResponse)
def get_session_gaze_report(
    session_id: str,
    actor: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
) -> InterviewGazeReportResponse:
    session = InterviewSessionService.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    assert_session_access(session, actor)
    _ensure_gaze_and_review_tables(db)

    report = (
        db.query(InterviewGazeReport)
        .filter(InterviewGazeReport.session_id == session_id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Gaze report not found for this session")
    return _gaze_response(report)


# ============================================================================
# HR Review / Shortlist Endpoints (Admin & Recruiter)
# ============================================================================


@router.get("/sessions/{session_id}/review", response_model=InterviewReviewResponse)
def get_session_review(
    session_id: str,
    _: RequestActor = Depends(require_staff),
    db: Session = Depends(get_db),
) -> InterviewReviewResponse:
    session = InterviewSessionService.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    review = (
        db.query(InterviewReview)
        .filter(InterviewReview.session_id == session_id)
        .first()
    )
    if not review:
        return InterviewReviewResponse(
            session_id=session_id,
            shortlisted=False,
            decision="pending",
            hr_notes=None,
        )
    return _review_response(review, session_id)


@router.put("/sessions/{session_id}/review", response_model=InterviewReviewResponse)
def upsert_session_review(
    session_id: str,
    payload: InterviewReviewRequest,
    actor: RequestActor = Depends(require_staff),
    db: Session = Depends(get_db),
) -> InterviewReviewResponse:
    session = InterviewSessionService.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    _ensure_gaze_and_review_tables(db)

    now = datetime.now(timezone.utc)
    try:
        review = (
            db.query(InterviewReview)
            .filter(InterviewReview.session_id == session_id)
            .first()
        )
        if review is None:
            review = InterviewReview(
                session_id=session_id,
                shortlisted=False,
                decision="pending",
                created_at=now,
            )
            db.add(review)

        if payload.shortlisted is not None:
            review.shortlisted = payload.shortlisted
            if payload.decision is None:
                if payload.shortlisted and review.decision == "pending":
                    review.decision = "shortlisted"
                elif not payload.shortlisted and review.decision == "shortlisted":
                    review.decision = "pending"

        if payload.decision is not None:
            review.decision = payload.decision
            if payload.shortlisted is None and payload.decision == "shortlisted":
                review.shortlisted = True

        if payload.hr_notes is not None:
            review.hr_notes = payload.hr_notes.strip() or None

        review.reviewed_by_user_id = actor.user_id
        review.reviewed_by_email = actor.email or None
        review.updated_at = now
        db.commit()
        db.refresh(review)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to save HR review: {exc}"
        ) from exc
    return _review_response(review, session_id)


# ============================================================================
# Evidence Summary, Evaluations, Expert Annotations, Context Extraction
# ============================================================================


@router.post(
    "/sessions/{session_id}/evidence-summary",
    response_model=InterviewEvidenceSummaryResponse,
)
def create_interview_evidence_summary(
    session_id: str,
    actor: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
) -> InterviewEvidenceSummaryResponse:
    session = InterviewSessionService.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    assert_session_access(session, actor)
    _ensure_gaze_and_review_tables(db)
    try:
        summary = InterviewEvidenceSummaryService.create_or_update(db, session)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to aggregate interview evidence: {exc}",
        ) from exc
    return _evidence_summary_response(summary)


@router.get(
    "/sessions/{session_id}/evidence-summary",
    response_model=InterviewEvidenceSummaryResponse,
)
def get_interview_evidence_summary(
    session_id: str,
    actor: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
) -> InterviewEvidenceSummaryResponse:
    session = InterviewSessionService.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    assert_session_access(session, actor)
    _ensure_gaze_and_review_tables(db)
    summary = InterviewEvidenceSummaryService.get(db, session_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Interview evidence summary not found")
    return _evidence_summary_response(summary)


@router.post(
    "/sessions/{session_id}/evaluations",
    response_model=CandidateEvaluationResponse,
    status_code=status.HTTP_201_CREATED,
)
def evaluate_candidate(
    session_id: str,
    payload: CandidateEvaluationRequest,
    actor: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
) -> CandidateEvaluationResponse:
    session = InterviewSessionService.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    assert_session_access(session, actor)
    _ensure_gaze_and_review_tables(db)
    try:
        evaluation = CandidateEvaluationService.create(db, session_id, payload.model_name)
    except ValueError as exc:
        db.rollback()
        status_code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except RuntimeError as exc:
        db.rollback()
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to save candidate evaluation: {exc}"
        ) from exc
    return _evaluation_response(evaluation)


@router.get(
    "/sessions/{session_id}/evaluations",
    response_model=list[CandidateEvaluationResponse],
)
def list_candidate_evaluations(
    session_id: str,
    actor: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
) -> list[CandidateEvaluationResponse]:
    session = InterviewSessionService.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    assert_session_access(session, actor)
    _ensure_gaze_and_review_tables(db)
    return [
        _evaluation_response(item)
        for item in CandidateEvaluationService.list_for_session(db, session_id)
    ]


@router.post(
    "/sessions/{session_id}/expert-annotations",
    response_model=ExpertAnnotationResponse,
)
def save_expert_annotation(
    session_id: str,
    payload: ExpertAnnotationRequest,
    _: RequestActor = Depends(require_staff),
    db: Session = Depends(get_db),
) -> ExpertAnnotationResponse:
    try:
        annotation = ExpertConsensusService.create_annotation(
            db,
            session_id,
            payload.expert_id,
            payload.scores.model_dump(mode="json"),
            payload.notes,
        )
    except ValueError as exc:
        db.rollback()
        status_code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to save expert annotation: {exc}"
        ) from exc
    return _annotation_response(annotation)


@router.get(
    "/sessions/{session_id}/expert-consensus",
    response_model=ExpertConsensusResponse,
)
def get_expert_consensus(
    session_id: str,
    _: RequestActor = Depends(require_staff),
    db: Session = Depends(get_db),
) -> ExpertConsensusResponse:
    if not InterviewSessionService.get(db, session_id):
        raise HTTPException(status_code=404, detail="Interview session not found")
    annotations = ExpertConsensusService.list_annotations(db, session_id)
    result = ExpertConsensusService.consensus_from_annotations(annotations)
    return ExpertConsensusResponse(session_id=session_id, **result)


def _extract_context(
    db: Session,
    cv_text: str,
    jd_text: str,
    jd_filename: str | None = None,
    allow_external_ai: bool = False,
) -> InterviewContextResponse:
    try:
        extraction_cv = (
            AdaptiveQuestionEngine.redact_personal_data(cv_text)
            if allow_external_ai
            else cv_text
        )
        extraction_jd = (
            AdaptiveQuestionEngine.redact_personal_data(jd_text)
            if allow_external_ai
            else jd_text
        )
        cv_profile = CvIngestService.extract_profile(
            db, extraction_cv, use_ai=allow_external_ai
        )
        job_profile = CvIngestService.build_uploaded_job_context(
            db,
            extraction_jd,
            jd_filename,
            use_ai=allow_external_ai,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return InterviewContextResponse(
        cv_profile=cv_profile,
        job_profile=job_profile,
        metadata=ContextExtractionMetadata(
            cv_character_count=len(cv_text),
            jd_character_count=len(jd_text),
        ),
    )


@router.post("/extract-context", response_model=InterviewContextResponse)
def extract_interview_context(
    payload: InterviewContextRequest,
    _: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
) -> InterviewContextResponse:
    """Extract CV and JD independently without computing a match score."""
    return _extract_context(
        db,
        payload.cv_text,
        payload.jd_text,
        allow_external_ai=payload.allow_external_ai,
    )


@router.post("/extract-context-files", response_model=InterviewContextResponse)
async def extract_interview_context_files(
    cv_file: UploadFile = File(..., description="CV: PDF, DOCX, TXT, or MD"),
    jd_file: UploadFile = File(..., description="JD: PDF, DOCX, TXT, MD, PNG, JPG, or WebP"),
    allow_external_ai: bool = Form(default=False),
    _: RequestActor = Depends(get_request_actor),
    db: Session = Depends(get_db),
) -> InterviewContextResponse:
    """Read uploaded CV/JD files and return structured interview context only."""
    try:
        cv_bytes = await cv_file.read()
        jd_bytes = await jd_file.read()
        cv_text = CvIngestService.extract_text_from_file(
            cv_file.filename or "cv",
            cv_file.content_type,
            cv_bytes,
        )
        jd_text = CvIngestService.extract_text_from_job_file(
            jd_file.filename or "jd",
            jd_file.content_type,
            jd_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to read uploaded files: {exc}"
        ) from exc

    return _extract_context(db, cv_text, jd_text, jd_file.filename, allow_external_ai)
