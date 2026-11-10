from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.config import BASE_DIR, settings

if TYPE_CHECKING:
    from app.services.question_ranking_v2_service import PairwiseQuestionRanker
    from app.services.weak_supervision_training_service import CandidateAnswerScorer


NON_ANSWER_PATTERNS = [
    r"^(?:dạ\s+|vâng\s+|à\s+|ừm\s+)?(?:tôi|em|mình)?\s*(?:không\s+biết|ko\s+biết|k\s+biết|chưa\s+biết|không\s+rõ|ko\s+rõ|k\s+rõ|chưa\s+rõ|chưa\s+học|chưa\s+tìm\s+hiểu|không\s+nắm\s+được|không\s+trả\s+lời\s+được|chưa\s+tiếp\s+cận|chưa\s+làm|chưa\s+từng\s+làm)(?:\s+(?:ạ|ạ\s*\.|nhiều|rõ|lắm|phần\s+này|câu\s+này|về\s+cái\s+này|về\s+phần\s+này))?[\.\s]*$",
    r"^(?:dạ\s+|vâng\s+)?(?:xin\s+phép\s+|cho\s+em\s+)?(?:bỏ\s+qua|bo\s+qua|cho\s+qua|qua\s+câu\s+này|bỏ\s+qua\s+câu\s+này)(?:\s+(?:ạ|câu\s+này|ạ\s*\.))?[\.\s]*$",
    r"^(?:dạ\s+)?(?:tôi|em)?\s*(?:chịu|chịu\s+thôi|chịu\s+ạ|không\s+có\s+câu\s+trả\s+lời)[\.\s]*$",
    r"^(?:pass|skip|idk|no\s+idea|no\s+answer|next)[\.\s!]*$",
    r"^[\.\s\?\-,\!_]+$",
]

REFUSAL_PHRASES = (
    "không biết",
    "ko biết",
    "k biết",
    "chưa biết",
    "không rõ",
    "ko rõ",
    "k rõ",
    "chưa rõ",
    "chưa tìm hiểu",
    "chưa học",
    "chưa tiếp cận",
    "chưa từng làm",
    "chưa làm qua",
    "chưa có kinh nghiệm về",
    "không trả lời được",
    "chưa trả lời được",
    "không nắm được",
    "không có câu trả lời",
    "bỏ qua",
    "bo qua",
    "xin phép bỏ qua",
    "cho qua",
    "cho em qua",
    "i don't know",
    "i do not know",
    "no idea",
    "pass",
    "skip",
)


def is_non_answer(text: str) -> bool:
    """Xác định câu trả lời có phải là không biết, từ chối hoặc bỏ qua không."""
    if not text:
        return True
    cleaned = text.strip().lower()
    if len(cleaned) == 0:
        return True

    # 1. Toàn dấu câu hoặc quá ngắn (< 3 ký tự)
    if re.match(r"^[\.\s\?\-,\!_]+$", cleaned) or len(cleaned) < 3:
        return True

    # 2. Khớp các mẫu câu từ chối cụ thể
    for pat in NON_ANSWER_PATTERNS:
        if re.search(pat, cleaned, re.IGNORECASE):
            return True

    # 3. Câu ngắn (dưới 18 từ) chứa cụm từ từ chối mà không có phần giải thích kỹ thuật
    words = cleaned.split()
    if len(words) <= 18:
        has_refusal = any(p in cleaned for p in REFUSAL_PHRASES)
        if has_refusal:
            has_contrast = any(
                c in cleaned
                for c in [
                    "nhưng",
                    "tuy nhiên",
                    "song",
                    "theo em",
                    "em nghĩ",
                    "thực ra là",
                    "thực tế là",
                ]
            )
            if not has_contrast:
                return True

    return False


class InterviewModelService:
    """Lazy, process-local registry for trained realtime interview models."""

    _lock = threading.RLock()
    _ranker: PairwiseQuestionRanker | None = None
    _ranker_path: Path | None = None
    _ranker_error: str | None = None
    _answer_scorer: CandidateAnswerScorer | None = None
    _answer_scorer_path: Path | None = None
    _answer_scorer_error: str | None = None

    @staticmethod
    def _resolve_path(raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        return path if path.is_absolute() else BASE_DIR / path

    @classmethod
    def reset_cache(cls) -> None:
        with cls._lock:
            cls._ranker = None
            cls._ranker_path = None
            cls._ranker_error = None
            cls._answer_scorer = None
            cls._answer_scorer_path = None
            cls._answer_scorer_error = None

    @classmethod
    def _get_ranker(cls) -> PairwiseQuestionRanker:
        from app.services.question_ranking_v2_service import PairwiseQuestionRanker

        path = cls._resolve_path(settings.INTERVIEW_QUESTION_RANKER_ARTIFACT)
        with cls._lock:
            if cls._ranker is not None and cls._ranker_path == path:
                return cls._ranker
            try:
                cls._ranker = PairwiseQuestionRanker(path)
                cls._ranker_path = path
                cls._ranker_error = None
                return cls._ranker
            except Exception as exc:
                cls._ranker = None
                cls._ranker_path = path
                cls._ranker_error = str(exc)[:500]
                raise

    @classmethod
    def _get_answer_scorer(cls) -> CandidateAnswerScorer:
        from app.services.weak_supervision_training_service import CandidateAnswerScorer

        path = cls._resolve_path(settings.INTERVIEW_ANSWER_SCORER_ARTIFACT)
        with cls._lock:
            if cls._answer_scorer is not None and cls._answer_scorer_path == path:
                return cls._answer_scorer
            try:
                cls._answer_scorer = CandidateAnswerScorer(path)
                cls._answer_scorer_path = path
                cls._answer_scorer_error = None
                return cls._answer_scorer
            except Exception as exc:
                cls._answer_scorer = None
                cls._answer_scorer_path = path
                cls._answer_scorer_error = str(exc)[:500]
                raise

    @classmethod
    def rank_questions(
        cls,
        context_text: str,
        questions: list[str],
        stage: str,
        target_criterion: str,
    ) -> tuple[list[float], str]:
        ranker = cls._get_ranker()
        rows = [
            {
                "context_text": context_text,
                "stage": stage,
                "target_criterion": target_criterion,
                "question_text": question,
            }
            for question in questions
        ]
        scores = ranker.score_rows(rows)
        model_type = str(ranker.artifact.get("model_type") or "unknown")
        return [round(float(score), 6) for score in scores], model_type

    @classmethod
    def score_answer(
        cls,
        context_text: str,
        question: str,
        answer: str,
        stage: str,
        target_criterion: str,
    ) -> tuple[dict[str, float], str]:
        # Xử lý trường hợp không trả lời, bỏ qua hoặc câu trả lời không biết
        if is_non_answer(answer):
            criteria = (
                "answer_relevance",
                "technical_correctness",
                "expected_signal_coverage",
                "reasoning_quality",
                "evidence_quality",
                "clarity_conciseness",
                "consistency",
            )
            # Điểm sàn tối thiểu 1.0/5.0 cho việc không cung cấp bằng chứng năng lực
            return {criterion: 1.0 for criterion in criteria}, "non_answer_fallback"

        scorer = cls._get_answer_scorer()
        scores = scorer.score(
            context_text,
            question,
            answer,
            stage,
            target_criterion,
        )
        model_type = str(scorer.artifact.get("model_type") or "unknown")
        return scores, model_type

    @classmethod
    def status(cls, load_models: bool = True) -> dict[str, Any]:
        if load_models:
            for loader in (cls._get_ranker, cls._get_answer_scorer):
                try:
                    loader()
                except Exception:
                    pass
        ranker_path = cls._resolve_path(settings.INTERVIEW_QUESTION_RANKER_ARTIFACT)
        scorer_path = cls._resolve_path(settings.INTERVIEW_ANSWER_SCORER_ARTIFACT)
        configured_providers = []
        from app.services.llm_provider_service import LLMProviderService

        for spec in LLMProviderService.pool_specs():
            if LLMProviderService.is_configured(spec):
                configured_providers.append(spec.provider)
        return {
            "multi_provider_enabled": settings.INTERVIEW_MULTI_PROVIDER_ENABLED,
            "question_candidate_count": max(
                1, min(4, settings.INTERVIEW_QUESTION_CANDIDATE_COUNT)
            ),
            "configured_providers": configured_providers,
            "question_ranker": {
                "ready": cls._ranker is not None,
                "model_type": (
                    cls._ranker.artifact.get("model_type") if cls._ranker else None
                ),
                "artifact_path": str(ranker_path),
                "error": cls._ranker_error,
            },
            "answer_scorer": {
                "ready": cls._answer_scorer is not None,
                "model_type": (
                    cls._answer_scorer.artifact.get("model_type")
                    if cls._answer_scorer
                    else None
                ),
                "artifact_path": str(scorer_path),
                "error": cls._answer_scorer_error,
            },
        }
