from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.interview_session import InterviewSession
from app.models.interview_turn import InterviewTurn
from app.research.rubrics import CANDIDATE_EVALUATION_RUBRIC
from app.services.llm_provider_service import (
    LLMJsonResult,
    LLMProviderError,
    LLMProviderSpec,
    LLMProviderService,
)
from app.services.interview_model_service import InterviewModelService
from app.services.interview_evidence_aggregation_service import (
    InterviewEvidenceSummaryService,
)
from app.services.pdf_processor import CvIngestService
from app.services.tts_prewarm_service import tts_prewarm_service


PROMPT_VERSION = "interview-question-v1"
VALID_CRITERIA = tuple(CANDIDATE_EVALUATION_RUBRIC["criteria"])


@dataclass(frozen=True)
class GeneratedQuestion:
    stage: str
    criterion: str
    question: str
    expected_signal: str
    source: str
    model_name: str | None
    prompt_hash: str
    latency_ms: int
    provider: str | None = None
    question_type: str = "unspecified"
    difficulty: int = 3
    required_signals: tuple[str, ...] = ()
    bonus_signals: tuple[str, ...] = ()
    applicable_answer_criteria: tuple[str, ...] = ()
    contradiction_rules: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuestionSelection:
    selected: GeneratedQuestion
    candidates: tuple[dict[str, Any], ...]
    selected_score: float | None
    ranker_version: str | None
    selection_latency_ms: int


class AdaptiveQuestionEngine:
    JSON_MAX_ATTEMPTS = 3
    QUESTION_TYPES = (
        "factual",
        "technical_scenario",
        "problem_solving",
        "project_evidence",
        "behavioral",
        "role_readiness",
        "follow_up",
    )
    TURN_EVIDENCE_CRITERIA = (
        "answer_relevance",
        "technical_correctness",
        "expected_signal_coverage",
        "reasoning_quality",
        "evidence_quality",
        "clarity_conciseness",
        "consistency",
    )

    @staticmethod
    def redact_personal_data(text: str) -> str:
        value = text or ""
        value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", value)
        value = re.sub(r"https?://\S+|www\.\S+", "[REDACTED_URL]", value, flags=re.IGNORECASE)
        value = re.sub(
            r"(?<!\d)(?:\+?84|0)(?:[ .-]?\d){8,10}(?!\d)",
            "[REDACTED_PHONE]",
            value,
        )
        return value

    @staticmethod
    def scheduled_stage(turn_index: int, max_questions: int) -> str:
        if turn_index == 0:
            return "introduction"
        if turn_index == max_questions - 1:
            return "closing"
        progress = turn_index / max(1, max_questions - 1)
        if progress < 0.3:
            return "cv_evidence"
        if progress < 0.55:
            return "technical"
        if progress < 0.8:
            return "problem_solving"
        return "behavioral"

    @classmethod
    def determine_stage(
        cls,
        turn_index: int,
        max_questions: int,
        history: list[dict[str, Any]],
    ) -> str:
        scheduled = cls.scheduled_stage(turn_index, max_questions)
        if scheduled == "closing" or not history:
            return scheduled
        previous = history[-1]
        answer = str(previous.get("answer") or "").strip()
        if previous.get("stage") not in {"introduction", "closing"} and len(answer.split()) < 20:
            return "follow_up"
        return scheduled

    @staticmethod
    def criterion_for_stage(stage: str, history: list[dict[str, Any]]) -> str:
        if stage == "follow_up" and history:
            previous = str(history[-1].get("criterion") or "")
            if previous in VALID_CRITERIA:
                return previous
        return {
            "introduction": "communication",
            "cv_evidence": "experience_evidence",
            "technical": "technical_knowledge",
            "problem_solving": "problem_solving",
            "behavioral": "learning_adaptability",
            "closing": "overall_readiness",
        }.get(stage, "role_relevance")

    @classmethod
    def build_prompt(
        cls,
        stage: str,
        criterion: str,
        language: str,
        cv_text: str,
        jd_text: str,
        cv_profile: dict[str, Any],
        job_profile: dict[str, Any],
        history: list[dict[str, Any]],
        generation_controls: dict[str, Any] | None = None,
    ) -> str:
        recent_history = [
            {
                "stage": item.get("stage"),
                "criterion": item.get("criterion"),
                "question": cls.redact_personal_data(str(item.get("question") or ""))[:800],
                "answer": cls.redact_personal_data(str(item.get("answer") or ""))[:1600],
            }
            for item in history[-4:]
        ]
        context = {
            "cv_profile": cv_profile,
            "job_profile": job_profile,
            "cv_excerpt": cls.redact_personal_data(cv_text)[:5000],
            "jd_excerpt": cls.redact_personal_data(jd_text)[:5000],
            "recent_history": recent_history,
            "next_stage": stage,
            "target_competency": criterion,
        }
        output_language = "Vietnamese" if language == "vi" else "English"
        controls = generation_controls or {}
        if controls:
            context["generation_controls"] = controls
            output_schema = (
                '{"question": string, "criterion": string, "expected_signal": string, '
                '"question_type": string, "difficulty": integer, '
                '"required_signals": [string], "bonus_signals": [string], '
                '"applicable_answer_criteria": [string], '
                '"contradiction_rules": [string]}\n'
                f"question_type must be one of: {', '.join(cls.QUESTION_TYPES)}. "
                "difficulty must be an integer from 1 through 5. "
                "required_signals must contain observable minimum-sufficient answer points. "
                "Only include answer criteria that the question can fairly elicit. "
                f"Applicable answer criteria are: {', '.join(cls.TURN_EVIDENCE_CRITERIA)}.\n"
            )
        else:
            output_schema = '{"question": string, "criterion": string, "expected_signal": string}\n'
        return (
            "You are conducting a structured, job-related technical interview. "
            "Generate exactly one adaptive question from the supplied context. "
            "Do not score the candidate, repeat a question, or ask about protected personal characteristics. "
            "For follow_up, probe the previous answer for evidence or technical reasoning. "
            "Return valid JSON only, without markdown.\n"
            f"Question language: {output_language}.\n"
            f"{output_schema}"
            f"criterion must be one of: {', '.join(VALID_CRITERIA)}.\n"
            f"Context:\n{json.dumps(context, ensure_ascii=False)}"
        )

    @staticmethod
    def fallback_question(
        stage: str,
        criterion: str,
        language: str,
        cv_profile: dict[str, Any],
        job_profile: dict[str, Any],
    ) -> tuple[str, str]:
        cv_skills = cv_profile.get("cv_skills") or []
        job_skills = job_profile.get("job_skills") or []
        cv_skill = str(cv_skills[0].get("name")) if cv_skills else "một công nghệ trong CV"
        job_skill = str(job_skills[0].get("name")) if job_skills else "một yêu cầu chính của vị trí"
        if language == "vi":
            questions = {
                "introduction": "Bạn hãy giới thiệu ngắn gọn về kinh nghiệm phù hợp nhất với vị trí này.",
                "cv_evidence": f"Hãy mô tả một dự án bạn trực tiếp sử dụng {cv_skill}, vai trò của bạn và kết quả đo lường được.",
                "technical": f"Với {job_skill}, bạn sẽ cân nhắc những lựa chọn kỹ thuật và đánh đổi quan trọng nào?",
                "problem_solving": f"Nếu hệ thống dùng {job_skill} gặp lỗi trên production, bạn sẽ điều tra và xử lý theo trình tự nào?",
                "behavioral": "Hãy kể về một lần yêu cầu dự án thay đổi đáng kể và cách bạn thích nghi.",
                "closing": "Bạn muốn bổ sung bằng chứng nào khác để thể hiện mức độ sẵn sàng cho vị trí này?",
                "follow_up": "Câu trả lời vừa rồi còn ngắn; bạn có thể nêu tình huống cụ thể, hành động và kết quả không?",
            }
        else:
            questions = {
                "introduction": "Please briefly introduce the experience most relevant to this role.",
                "cv_evidence": f"Describe a project where you directly used {cv_skill}, your role, and a measurable result.",
                "technical": f"What technical choices and trade-offs would you consider when using {job_skill}?",
                "problem_solving": f"If a production system using {job_skill} failed, how would you investigate and resolve it?",
                "behavioral": "Describe a significant project requirement change and how you adapted to it.",
                "closing": "What additional evidence would you like to share about your readiness for this role?",
                "follow_up": "Your previous answer was brief. Could you give a concrete situation, your actions, and the outcome?",
            }
        return questions.get(stage, questions["technical"]), f"Observable evidence for {criterion.replace('_', ' ')}."

    @classmethod
    def generate(
        cls,
        turn_index: int,
        max_questions: int,
        language: str,
        cv_text: str,
        jd_text: str,
        cv_profile: dict[str, Any],
        job_profile: dict[str, Any],
        history: list[dict[str, Any]],
        allow_external_ai: bool,
        model_name: str | None = None,
        provider: str | None = None,
        strict_external_ai: bool = False,
        generation_controls: dict[str, Any] | None = None,
    ) -> GeneratedQuestion:
        stage = cls.determine_stage(turn_index, max_questions, history)
        criterion = cls.criterion_for_stage(stage, history)
        prompt = cls.build_prompt(
            stage,
            criterion,
            language,
            cv_text,
            jd_text,
            cv_profile,
            job_profile,
            history,
            generation_controls,
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        started_at = time.perf_counter()
        data: dict[str, Any] | None = None

        selected_identity = provider or model_name or "groq"
        selected_model = model_name
        selected_provider = provider
        if allow_external_ai and settings.CV_AI_ENRICHMENT_ENABLED:
            try:
                response: LLMJsonResult | None = None
                correction = ""
                for attempt in range(1, cls.JSON_MAX_ATTEMPTS + 1):
                    effective_prompt = prompt + correction
                    try:
                        response = LLMProviderService.generate_json(
                            selected_identity,
                            effective_prompt,
                            operation="question_generation",
                            max_output_tokens=1024,
                            temperature=0.3,
                        )
                        prompt_hash = hashlib.sha256(
                            effective_prompt.encode("utf-8")
                        ).hexdigest()
                    except LLMProviderError as exc:
                        if exc.error_code != "invalid_json" or attempt >= cls.JSON_MAX_ATTEMPTS:
                            raise
                        correction = (
                            "\nYour previous response was not valid JSON. Return exactly one "
                            "complete JSON object matching every requested output field. "
                            "Do not use markdown or explanatory text."
                        )
                        continue

                    candidate_data = response.data
                    candidate_question = str(
                        candidate_data.get("question") or ""
                    ).strip()
                    if candidate_question and len(candidate_question) <= 1200:
                        data = candidate_data
                        selected_model = response.model
                        selected_provider = response.provider
                        break
                    if attempt >= cls.JSON_MAX_ATTEMPTS:
                        if strict_external_ai:
                            raise LLMProviderError(
                                response.provider,
                                f"{response.provider} did not generate a valid question "
                                f"after {cls.JSON_MAX_ATTEMPTS} attempts",
                                error_code="invalid_content",
                            )
                        break
                    correction = (
                        "\nYour previous JSON object had a missing, empty, or oversized "
                        "question field. Return exactly one complete JSON object with a "
                        "non-empty question of at most 1200 characters and every requested "
                        "output field. Do not use markdown or explanatory text."
                    )
                if response is None:
                    raise RuntimeError("Question generator returned no response")
            except Exception as exc:
                if strict_external_ai:
                    raise
                print(f"--- INTERVIEW QUESTION AI ERROR: {exc} ---")

        latency_ms = max(0, round((time.perf_counter() - started_at) * 1000))
        if data:
            question = str(data.get("question") or "").strip()
            generated_criterion = str(data.get("criterion") or "").strip()
            expected_signal = str(data.get("expected_signal") or "").strip()
            if question and len(question) <= 1200:
                if (
                    generated_criterion in VALID_CRITERIA
                    and not (generation_controls or {}).get("lock_criterion")
                ):
                    criterion = generated_criterion
                question_type = str(data.get("question_type") or "").strip()
                if question_type not in cls.QUESTION_TYPES:
                    question_type = cls._default_question_type(stage)
                try:
                    difficulty = int(data.get("difficulty") or 3)
                except (TypeError, ValueError):
                    difficulty = 3
                requested_difficulty = (generation_controls or {}).get("difficulty")
                if requested_difficulty is not None:
                    try:
                        difficulty = int(requested_difficulty)
                    except (TypeError, ValueError):
                        pass
                difficulty = min(5, max(1, difficulty))
                required_signals = cls._string_tuple(data.get("required_signals"), 8)
                if not required_signals and expected_signal:
                    required_signals = (expected_signal[:1200],)
                bonus_signals = cls._string_tuple(data.get("bonus_signals"), 6)
                applicable = tuple(
                    item
                    for item in cls._string_tuple(
                        data.get("applicable_answer_criteria"),
                        len(cls.TURN_EVIDENCE_CRITERIA),
                    )
                    if item in cls.TURN_EVIDENCE_CRITERIA
                )
                required_applicability = set(
                    cls._default_applicable_criteria(question_type)
                )
                required_applicability.update(applicable)
                applicable = tuple(
                    criterion
                    for criterion in cls.TURN_EVIDENCE_CRITERIA
                    if criterion in required_applicability
                )
                contradiction_rules = cls._string_tuple(
                    data.get("contradiction_rules"),
                    6,
                )
                return GeneratedQuestion(
                    stage, criterion, question, expected_signal[:1200], "ai",
                    selected_model, prompt_hash, latency_ms, selected_provider,
                    question_type, difficulty, required_signals, bonus_signals,
                    applicable, contradiction_rules,
                )

        question, expected_signal = cls.fallback_question(stage, criterion, language, cv_profile, job_profile)
        return GeneratedQuestion(
            stage,
            criterion,
            question,
            expected_signal,
            "fallback",
            None,
            prompt_hash,
            latency_ms,
            question_type=cls._default_question_type(stage),
            difficulty=min(5, max(1, int((generation_controls or {}).get("difficulty", 3)))),
            required_signals=(expected_signal,),
            applicable_answer_criteria=cls._default_applicable_criteria(
                cls._default_question_type(stage)
            ),
        )

    @staticmethod
    def _string_tuple(value: Any, limit: int) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        items: list[str] = []
        for raw in value:
            item = str(raw or "").strip()
            if item and item not in items:
                items.append(item[:800])
            if len(items) >= limit:
                break
        return tuple(items)

    @staticmethod
    def _default_question_type(stage: str) -> str:
        return {
            "introduction": "role_readiness",
            "cv_evidence": "project_evidence",
            "technical": "technical_scenario",
            "problem_solving": "problem_solving",
            "behavioral": "behavioral",
            "closing": "role_readiness",
            "follow_up": "follow_up",
        }.get(stage, "technical_scenario")

    @classmethod
    def _default_applicable_criteria(cls, question_type: str) -> tuple[str, ...]:
        always = (
            "answer_relevance",
            "expected_signal_coverage",
            "clarity_conciseness",
            "consistency",
        )
        if question_type == "factual":
            return always[:1] + ("technical_correctness",) + always[1:]
        if question_type in {"behavioral", "project_evidence"}:
            return always[:2] + (
                "reasoning_quality",
                "evidence_quality",
            ) + always[2:]
        return cls.TURN_EVIDENCE_CRITERIA

    @staticmethod
    def model_context(
        cv_profile: dict[str, Any],
        job_profile: dict[str, Any],
        history: list[dict[str, Any]],
        stage: str,
        criterion: str,
    ) -> str:
        return json.dumps(
            {
                "cv_profile": cv_profile,
                "job_profile": job_profile,
                "interview_history": history[-4:],
                "stage": stage,
                "target_criterion": criterion,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _candidate_payload(generated: GeneratedQuestion, status: str = "success") -> dict[str, Any]:
        return {
            "status": status,
            "provider": generated.provider,
            "model_name": generated.model_name,
            "stage": generated.stage,
            "criterion": generated.criterion,
            "question": generated.question,
            "expected_signal": generated.expected_signal,
            "question_type": generated.question_type,
            "difficulty": generated.difficulty,
            "required_signals": list(generated.required_signals),
            "bonus_signals": list(generated.bonus_signals),
            "applicable_answer_criteria": list(generated.applicable_answer_criteria),
            "contradiction_rules": list(generated.contradiction_rules),
            "generation_latency_ms": generated.latency_ms,
            "prompt_hash": generated.prompt_hash,
            "pairwise_utility": None,
            "selected": False,
        }

    @classmethod
    def generate_ranked(
        cls,
        turn_index: int,
        max_questions: int,
        language: str,
        cv_text: str,
        jd_text: str,
        cv_profile: dict[str, Any],
        job_profile: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> QuestionSelection:
        started_at = time.perf_counter()
        target_count = max(1, min(4, settings.INTERVIEW_QUESTION_CANDIDATE_COUNT))
        configured = [
            spec
            for spec in LLMProviderService.pool_specs()
            if LLMProviderService.is_configured(spec)
        ]
        if configured:
            offset = turn_index % len(configured)
            configured = configured[offset:] + configured[:offset]

        successful: list[tuple[GeneratedQuestion, dict[str, Any]]] = []
        attempts: list[dict[str, Any]] = []
        seen_questions: set[str] = set()

        def generate_one(spec: LLMProviderSpec, slot: int) -> GeneratedQuestion:
            return cls.generate(
                turn_index,
                max_questions,
                language,
                cv_text,
                jd_text,
                cv_profile,
                job_profile,
                history,
                True,
                model_name=spec.model,
                provider=spec.provider,
                strict_external_ai=True,
                generation_controls={
                    "candidate_slot": slot,
                    "candidate_count": target_count,
                    "lock_criterion": True,
                    "diversity_instruction": (
                        "Generate a distinct, evidence-seeking question for this context."
                    ),
                },
            )

        def run_batch(specs: list[LLMProviderSpec], slot_start: int) -> None:
            if not specs:
                return
            with ThreadPoolExecutor(max_workers=len(specs)) as executor:
                futures = {
                    spec.provider: executor.submit(
                        generate_one,
                        spec,
                        slot_start + position,
                    )
                    for position, spec in enumerate(specs)
                }
                for spec in specs:
                    try:
                        generated = futures[spec.provider].result()
                        payload = cls._candidate_payload(generated)
                        normalized = generated.question.strip().casefold()
                        if normalized in seen_questions:
                            payload["status"] = "duplicate"
                            attempts.append(payload)
                            continue
                        seen_questions.add(normalized)
                        attempts.append(payload)
                        successful.append((generated, payload))
                    except Exception as exc:
                        attempts.append(
                            {
                                "status": "failed",
                                "provider": spec.provider,
                                "model_name": spec.model,
                                "error_code": getattr(exc, "error_code", "generation_error"),
                                "quota_exhausted": bool(
                                    getattr(exc, "quota_exhausted", False)
                                ),
                                "error": str(exc)[:500],
                                "selected": False,
                                "pairwise_utility": None,
                            }
                        )

        run_batch(configured[:target_count], 1)
        if len(successful) < target_count:
            for spec in configured[target_count:]:
                run_batch([spec], len(attempts) + 1)
                if len(successful) >= target_count:
                    break

        if not successful:
            fallback = cls.generate(
                turn_index,
                max_questions,
                language,
                cv_text,
                jd_text,
                cv_profile,
                job_profile,
                history,
                False,
            )
            payload = cls._candidate_payload(fallback, "fallback")
            payload["selected"] = True
            attempts.append(payload)
            return QuestionSelection(
                selected=fallback,
                candidates=tuple(attempts),
                selected_score=None,
                ranker_version=None,
                selection_latency_ms=max(
                    0, round((time.perf_counter() - started_at) * 1000)
                ),
            )

        stage = successful[0][0].stage
        criterion = successful[0][0].criterion
        context_text = cls.model_context(
            cv_profile,
            job_profile,
            history,
            stage,
            criterion,
        )
        try:
            scores, ranker_version = InterviewModelService.rank_questions(
                context_text,
                [item[0].question for item in successful],
                stage,
                criterion,
            )
        except Exception as exc:
            scores = [0.0 for _item in successful]
            ranker_version = "fallback_provider_order_v1"
            attempts.append(
                {
                    "status": "ranker_failed",
                    "error": str(exc)[:500],
                    "selected": False,
                    "pairwise_utility": None,
                }
            )

        selected_index = max(range(len(scores)), key=scores.__getitem__)
        selected = successful[selected_index][0]
        for index, (_candidate, payload) in enumerate(successful):
            payload["pairwise_utility"] = scores[index]
            payload["selected"] = index == selected_index
        return QuestionSelection(
            selected=selected,
            candidates=tuple(attempts),
            selected_score=scores[selected_index],
            ranker_version=ranker_version,
            selection_latency_ms=max(
                0, round((time.perf_counter() - started_at) * 1000)
            ),
        )


class InterviewSessionService:
    @staticmethod
    def history(session: InterviewSession) -> list[dict[str, Any]]:
        return [
            {
                "stage": turn.stage,
                "criterion": turn.criterion,
                "question": turn.question_text,
                "answer": turn.answer_text,
            }
            for turn in session.turns
            if turn.answer_text is not None
        ]

    @staticmethod
    def new_turn(session: InterviewSession, turn_index: int) -> InterviewTurn:
        history = InterviewSessionService.history(session)
        if (
            session.allow_external_ai
            and settings.CV_AI_ENRICHMENT_ENABLED
            and settings.INTERVIEW_MULTI_PROVIDER_ENABLED
        ):
            selection = AdaptiveQuestionEngine.generate_ranked(
                turn_index,
                session.max_questions,
                session.language,
                session.cv_text,
                session.jd_text,
                session.cv_profile_json,
                session.job_profile_json,
                history,
            )
        else:
            generated = AdaptiveQuestionEngine.generate(
                turn_index,
                session.max_questions,
                session.language,
                session.cv_text,
                session.jd_text,
                session.cv_profile_json,
                session.job_profile_json,
                history,
                session.allow_external_ai,
            )
            payload = AdaptiveQuestionEngine._candidate_payload(generated)
            payload["selected"] = True
            selection = QuestionSelection(
                selected=generated,
                candidates=(payload,),
                selected_score=None,
                ranker_version=None,
                selection_latency_ms=generated.latency_ms,
            )
        generated = selection.selected
        return InterviewTurn(
            session_id=session.session_id,
            turn_index=turn_index,
            stage=generated.stage,
            criterion=generated.criterion,
            question_text=generated.question,
            expected_signal=generated.expected_signal,
            question_source=(
                "ai_ranked"
                if generated.source == "ai" and selection.ranker_version
                else generated.source
            ),
            provider_name=generated.provider,
            model_name=generated.model_name,
            prompt_version=PROMPT_VERSION,
            prompt_hash=generated.prompt_hash,
            generation_latency_ms=generated.latency_ms,
            question_candidates_json=list(selection.candidates),
            selected_question_score=selection.selected_score,
            question_ranker_version=selection.ranker_version,
            question_selection_latency_ms=selection.selection_latency_ms,
        )

    @staticmethod
    def _applicable_criteria(turn: InterviewTurn) -> list[str]:
        for candidate in turn.question_candidates_json or []:
            if candidate.get("selected"):
                criteria = candidate.get("applicable_answer_criteria") or []
                return [
                    str(item)
                    for item in criteria
                    if item in AdaptiveQuestionEngine.TURN_EVIDENCE_CRITERIA
                ]
        return list(
            AdaptiveQuestionEngine._default_applicable_criteria(
                AdaptiveQuestionEngine._default_question_type(turn.stage)
            )
        )

    @staticmethod
    def _score_turn_answer(
        session: InterviewSession,
        turn: InterviewTurn,
        answer: str,
        history_before_answer: list[dict[str, Any]],
    ) -> None:
        started_at = time.perf_counter()
        applicable = InterviewSessionService._applicable_criteria(turn)
        context_text = AdaptiveQuestionEngine.model_context(
            session.cv_profile_json,
            session.job_profile_json,
            history_before_answer,
            turn.stage,
            turn.criterion,
        )
        try:
            scores, scorer_version = InterviewModelService.score_answer(
                context_text,
                turn.question_text,
                answer,
                turn.stage,
                turn.criterion,
            )
            turn.answer_scorer_version = scorer_version
            turn.answer_scores_json = {
                "status": "success",
                "scope": "turn_evidence_only",
                "scores": scores,
                "applicable_criteria": applicable,
                "applicable_scores": {
                    criterion: scores[criterion]
                    for criterion in applicable
                    if criterion in scores
                },
            }
        except Exception as exc:
            turn.answer_scorer_version = None
            turn.answer_scores_json = {
                "status": "unavailable",
                "scope": "turn_evidence_only",
                "scores": {},
                "applicable_criteria": applicable,
                "applicable_scores": {},
                "error": str(exc)[:500],
            }
        turn.answer_scoring_latency_ms = max(
            0, round((time.perf_counter() - started_at) * 1000)
        )

    @staticmethod
    def create(
        db: Session,
        cv_text: str,
        jd_text: str,
        language: str,
        max_questions: int,
        allow_external_ai: bool,
        research_consent: bool,
        user_id: int | None = None,
        context_source: str = "user_supplied",
        source_cv_id: int | None = None,
        source_job_id: int | None = None,
    ) -> InterviewSession:
        extraction_cv_text = AdaptiveQuestionEngine.redact_personal_data(cv_text) if allow_external_ai else cv_text
        extraction_jd_text = AdaptiveQuestionEngine.redact_personal_data(jd_text) if allow_external_ai else jd_text
        cv_profile = CvIngestService.extract_profile(
            db,
            extraction_cv_text,
            use_ai=False,
        )
        job_profile = CvIngestService.build_uploaded_job_context(
            db,
            extraction_jd_text,
            use_ai=False,
        )
        session = InterviewSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            source_cv_id=source_cv_id,
            source_job_id=source_job_id,
            language=language,
            max_questions=max_questions,
            allow_external_ai=allow_external_ai,
            research_consent=research_consent,
            context_source=context_source,
            cv_text=cv_text.strip(),
            jd_text=jd_text.strip(),
            cv_profile_json=cv_profile.model_dump(mode="json"),
            job_profile_json=job_profile.model_dump(mode="json"),
        )
        db.add(session)
        db.flush()
        first_turn = InterviewSessionService.new_turn(session, 0)
        tts_prewarm_service.prewarm(first_turn.question_text)
        session.turns.append(first_turn)
        session.current_stage = first_turn.stage
        db.commit()
        db.refresh(session)
        return session

    @staticmethod
    def list_for_user(db: Session, user_id: int, limit: int = 30) -> list[InterviewSession]:
        return (
            db.query(InterviewSession)
            .filter(InterviewSession.user_id == user_id)
            .order_by(InterviewSession.created_at.desc())
            .limit(limit)
            .all()
        )

    @staticmethod
    def get(db: Session, session_id: str, lock: bool = False) -> InterviewSession | None:
        query = db.query(InterviewSession).filter(InterviewSession.session_id == session_id)
        return query.with_for_update().first() if lock else query.first()

    @staticmethod
    def submit_answer(db: Session, session_id: str, answer: str) -> InterviewSession:
        session = InterviewSessionService.get(db, session_id, lock=True)
        if not session:
            raise ValueError("Interview session not found")
        if session.status != "active":
            raise ValueError("Interview session is already completed")

        current_turn = next(
            (turn for turn in session.turns if turn.turn_index == session.current_turn_index), None
        )
        if not current_turn or current_turn.answer_text is not None:
            raise ValueError("The current interview question cannot accept an answer")

        history_before_answer = InterviewSessionService.history(session)
        normalized = answer.strip()
        current_turn.answer_text = normalized
        current_turn.answer_word_count = len(normalized.split())
        current_turn.answered_at = datetime.now(timezone.utc)
        InterviewSessionService._score_turn_answer(
            session,
            current_turn,
            normalized,
            history_before_answer,
        )
        next_index = current_turn.turn_index + 1

        if next_index >= session.max_questions:
            session.status = "completed"
            session.current_stage = "completed"
            session.completed_at = datetime.now(timezone.utc)
        else:
            db.flush()
            next_turn = InterviewSessionService.new_turn(session, next_index)
            tts_prewarm_service.prewarm(next_turn.question_text)
            session.turns.append(next_turn)
            session.current_turn_index = next_index
            session.current_stage = next_turn.stage

        session.updated_at = datetime.now(timezone.utc)
        if session.status == "completed":
            InterviewEvidenceSummaryService.create_or_update(
                db,
                session,
                commit=False,
            )
        db.commit()
        db.refresh(session)
        return session
