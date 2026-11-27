from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.schemas.cv import ExtractedCvProfile, JobContext


class InterviewContextRequest(BaseModel):
    cv_text: str = Field(..., min_length=30)
    jd_text: str = Field(..., min_length=30)
    allow_external_ai: bool = False


class ContextExtractionMetadata(BaseModel):
    cv_character_count: int = Field(..., ge=0)
    jd_character_count: int = Field(..., ge=0)
    extraction_version: str = "v1"


class InterviewContextResponse(BaseModel):
    cv_profile: ExtractedCvProfile
    job_profile: JobContext
    metadata: ContextExtractionMetadata


class InterviewSessionCreateRequest(InterviewContextRequest):
    user_id: int | None = Field(default=None, ge=1)
    language: str = "vi"
    max_questions: int = Field(default=8, ge=5, le=15)
    research_consent: bool = False

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"vi", "en"}:
            raise ValueError("language must be 'vi' or 'en'")
        return normalized


class SyntheticInterviewSessionCreateRequest(BaseModel):
    seed: int = Field(default=1, ge=0, le=2_147_483_647)
    role: str | None = Field(default=None, pattern="^(backend|frontend|data|devops|qa|mobile)$")
    level: str | None = Field(default=None, pattern="^(intern|junior|middle|senior)$")
    language: str = Field(default="vi", pattern="^(vi|en)$")
    max_questions: int = Field(default=5, ge=5, le=15)
    allow_external_ai: bool = False


class StoredInterviewSessionCreateRequest(BaseModel):
    cv_id: int = Field(..., ge=1)
    job_id: int = Field(..., ge=1)
    language: str = Field(default="vi", pattern="^(vi|en)$")
    max_questions: int = Field(default=8, ge=5, le=15)
    allow_external_ai: bool = True
    user_id: int | None = Field(default=None)


class StoredCvOption(BaseModel):
    cv_id: int
    file_name: str
    uploaded_at: datetime
    fallback_ready: bool


class StoredJobOption(BaseModel):
    job_id: int
    title: str
    company_name: str | None = None
    location: str | None = None
    level: str | None = None


class StoredInterviewOptionsResponse(BaseModel):
    cvs: list[StoredCvOption] = Field(default_factory=list)
    jobs: list[StoredJobOption] = Field(default_factory=list)


class InterviewAnswerRequest(BaseModel):
    answer: str = Field(..., min_length=2, max_length=10000)


class InterviewQuestionResponse(BaseModel):
    turn_index: int
    stage: str
    question: str


class InterviewQuestionCandidateResponse(BaseModel):
    provider_name: str | None = None
    model_name: str | None = None
    question: str
    criterion: str | None = None
    question_type: str | None = None
    difficulty: int | None = None
    pairwise_utility: float | None = None
    selected: bool = False
    generation_latency_ms: int = 0


class InterviewTurnResponse(InterviewQuestionResponse):
    criterion: str | None = None
    expected_signal: str | None = None
    question_type: str | None = None
    difficulty: int | None = None
    required_signals: list[str] = Field(default_factory=list)
    bonus_signals: list[str] = Field(default_factory=list)
    answer: str | None = None
    answer_word_count: int | None = None
    question_source: str
    provider_name: str | None = None
    model_name: str | None = None
    generation_latency_ms: int
    question_candidate_count: int = 1
    question_candidates: list[InterviewQuestionCandidateResponse] = Field(
        default_factory=list
    )
    selected_question_score: float | None = None
    question_ranker_version: str | None = None
    question_selection_latency_ms: int = 0
    answer_evidence: dict | None = None
    answer_scorer_version: str | None = None
    answer_scoring_latency_ms: int = 0
    created_at: datetime
    answered_at: datetime | None = None


class RealtimeInterviewModelStatusResponse(BaseModel):
    multi_provider_enabled: bool
    question_candidate_count: int
    configured_providers: list[str] = Field(default_factory=list)
    question_ranker: dict
    answer_scorer: dict


class InterviewEvidenceSummaryResponse(BaseModel):
    summary_id: int
    session_id: str
    aggregator_version: str
    status: str
    competencies: dict
    evidence: dict
    flags: list[dict] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class InterviewGazeEventPayload(BaseModel):
    timestamp: str
    direction: str
    duration: float = 0.0
    behavior: str | None = None
    penalty: float | None = None
    snapshot: str | None = None
    question_index: int | None = None
    question_text: str | None = None


class InterviewGazeReportRequest(BaseModel):
    total_frames: int = Field(default=0, ge=0)
    face_frames: int = Field(default=0, ge=0)
    duration_sec: float = Field(default=0.0, ge=0.0)
    camera_active: bool = False
    face_presence_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    gaze_center_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    gaze_left_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    gaze_right_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    gaze_away_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    gaze_events: list[InterviewGazeEventPayload] = Field(default_factory=list)
    gaze_timeline: list[dict] = Field(default_factory=list)
    integrity_score: float | None = Field(default=None, ge=0.0, le=100.0)
    integrity_rank: str | None = Field(default=None, pattern="^(HIGH|MEDIUM|LOW)$")
    integrity_summary: dict | None = None


class InterviewGazeReportResponse(InterviewGazeReportRequest):
    report_id: int
    session_id: str
    created_at: datetime
    updated_at: datetime


class InterviewReviewRequest(BaseModel):
    shortlisted: bool | None = None
    decision: str | None = Field(
        default=None,
        pattern="^(pending|shortlisted|interview_next|hired|rejected)$",
    )
    hr_notes: str | None = Field(default=None, max_length=5000)


class InterviewReviewResponse(BaseModel):
    review_id: int | None = None
    session_id: str
    shortlisted: bool = False
    decision: str = "pending"
    hr_notes: str | None = None
    reviewed_by_user_id: int | None = None
    reviewed_by_email: str | None = None
    updated_at: datetime | None = None


class InterviewSessionResponse(BaseModel):
    session_id: str
    user_id: int | None = None
    candidate_name: str | None = None
    candidate_email: str | None = None
    status: str
    language: str
    max_questions: int
    current_stage: str
    allow_external_ai: bool
    research_consent: bool
    context_source: str = "user_supplied"
    source_cv_id: int | None = None
    source_job_id: int | None = None
    cv_profile_json: dict | None = None
    job_profile_json: dict | None = None
    current_question: InterviewQuestionResponse | None = None
    turns: list[InterviewTurnResponse] = Field(default_factory=list)
    review: InterviewReviewResponse | None = None
    created_at: datetime
    completed_at: datetime | None = None


class CandidateRubricScores(BaseModel):
    technical_knowledge: float = Field(..., ge=1, le=5)
    problem_solving: float = Field(..., ge=1, le=5)
    experience_evidence: float = Field(..., ge=1, le=5)
    communication: float = Field(..., ge=1, le=5)
    role_relevance: float = Field(..., ge=1, le=5)
    learning_adaptability: float = Field(..., ge=1, le=5)
    overall_readiness: float = Field(..., ge=1, le=5)


class CandidateEvaluationRequest(BaseModel):
    model_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]+$",
    )


class CandidateEvaluationResponse(BaseModel):
    evaluation_id: int
    session_id: str
    provider: str
    model_name: str
    prompt_version: str
    scores: CandidateRubricScores
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    overall_explanation: str | None = None
    latency_ms: int
    created_at: datetime


class ExpertAnnotationRequest(BaseModel):
    expert_id: str = Field(..., min_length=2, max_length=100)
    scores: CandidateRubricScores
    notes: str | None = Field(default=None, max_length=5000)

    @field_validator("expert_id")
    @classmethod
    def normalize_expert_id(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 2:
            raise ValueError("expert_id must contain at least 2 non-space characters")
        return normalized


class ExpertAnnotationResponse(BaseModel):
    annotation_id: int
    session_id: str
    expert_id: str
    rubric_version: str
    scores: CandidateRubricScores
    notes: str | None = None
    created_at: datetime
    updated_at: datetime


class ConsensusCriterionStats(BaseModel):
    median: float
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float


class ExpertConsensusResponse(BaseModel):
    session_id: str
    status: str
    expert_count: int
    minimum_experts: int
    rubric_version: str
    reference_scores: CandidateRubricScores | None = None
    criterion_stats: dict[str, ConsensusCriterionStats] = Field(default_factory=dict)
    inter_rater_icc_2_k: float | None = None
