from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models.interview_evidence_summary import InterviewEvidenceSummary
from app.models.interview_session import InterviewSession
from app.services.interview_model_service import is_non_answer


AGGREGATOR_VERSION = "interview-evidence-v1"
BASE_COMPETENCIES = (
    "technical_knowledge",
    "problem_solving",
    "experience_evidence",
    "communication",
    "role_relevance",
    "learning_adaptability",
)
TURN_SCORE_WEIGHTS = {
    "answer_relevance": 0.15,
    "technical_correctness": 0.20,
    "expected_signal_coverage": 0.20,
    "reasoning_quality": 0.15,
    "evidence_quality": 0.15,
    "clarity_conciseness": 0.075,
    "consistency": 0.075,
}
COMPETENCY_SIGNALS = {
    "technical_knowledge": {
        "technical_correctness": 0.40,
        "expected_signal_coverage": 0.25,
        "reasoning_quality": 0.20,
        "answer_relevance": 0.10,
        "consistency": 0.05,
    },
    "problem_solving": {
        "reasoning_quality": 0.35,
        "technical_correctness": 0.25,
        "expected_signal_coverage": 0.20,
        "evidence_quality": 0.10,
        "consistency": 0.10,
    },
    "experience_evidence": {
        "evidence_quality": 0.40,
        "expected_signal_coverage": 0.25,
        "answer_relevance": 0.15,
        "consistency": 0.15,
        "clarity_conciseness": 0.05,
    },
    "communication": {
        "clarity_conciseness": 0.45,
        "answer_relevance": 0.30,
        "consistency": 0.15,
        "expected_signal_coverage": 0.10,
    },
    "role_relevance": {
        "expected_signal_coverage": 0.40,
        "answer_relevance": 0.30,
        "evidence_quality": 0.15,
        "consistency": 0.15,
    },
    "learning_adaptability": {
        "reasoning_quality": 0.30,
        "evidence_quality": 0.30,
        "expected_signal_coverage": 0.20,
        "answer_relevance": 0.10,
        "consistency": 0.10,
    },
}
OVERALL_WEIGHTS = {
    "technical_knowledge": 0.20,
    "problem_solving": 0.20,
    "experience_evidence": 0.15,
    "communication": 0.15,
    "role_relevance": 0.15,
    "learning_adaptability": 0.15,
}


class InterviewEvidenceAggregator:
    @staticmethod
    def _valid_scores(turn: Any) -> tuple[dict[str, float], set[str]] | None:
        payload = turn.answer_scores_json or {}
        ans = (turn.answer_text or "").strip()

        # Trường hợp ứng viên không trả lời, bỏ qua hoặc câu trả lời không biết
        if not ans or is_non_answer(ans) or payload.get("is_non_answer"):
            criteria = (
                "answer_relevance",
                "technical_correctness",
                "expected_signal_coverage",
                "reasoning_quality",
                "evidence_quality",
                "clarity_conciseness",
                "consistency",
            )
            scores = {c: 1.0 for c in criteria}
            applicable = {
                str(item) for item in payload.get("applicable_criteria") or []
            }
            if not applicable and getattr(turn, "criterion", None):
                applicable = {str(turn.criterion)}
            return scores, applicable or set(criteria)

        if payload.get("status") != "success":
            return None
        raw_scores = payload.get("scores") or {}
        scores: dict[str, float] = {}
        for name, raw_value in raw_scores.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and 1 <= value <= 5:
                scores[str(name)] = value
        applicable = {
            str(item) for item in payload.get("applicable_criteria") or []
        }
        return (scores, applicable) if scores and applicable else None

    @classmethod
    def _turn_competency_score(cls, turn: Any, competency: str) -> float | None:
        parsed = cls._valid_scores(turn)
        if not parsed:
            return None
        scores, applicable = parsed
        weights = COMPETENCY_SIGNALS.get(competency, TURN_SCORE_WEIGHTS)
        usable = {
            name: weight
            for name, weight in weights.items()
            if name in applicable and name in scores
        }
        total_weight = sum(usable.values())
        if not total_weight:
            return None
        return round(
            sum(scores[name] * weight for name, weight in usable.items())
            / total_weight,
            3,
        )

    @staticmethod
    def _eligible(turn: Any, competency: str) -> tuple[bool, float, str]:
        if turn.criterion == competency:
            return True, 1.0, "direct"
        if competency == "communication":
            return True, 0.6, "cross_turn_communication"
        if competency == "role_relevance":
            return True, 0.6, "cross_turn_role_relevance"
        return False, 0.0, "not_applicable"

    @classmethod
    def _aggregate_competency(
        cls,
        turns: list[Any],
        competency: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        evidence: list[dict[str, Any]] = []
        for turn in turns:
            eligible, evidence_weight, source = cls._eligible(turn, competency)
            if not eligible:
                continue
            score = cls._turn_competency_score(turn, competency)
            if score is None:
                continue
            payload = turn.answer_scores_json or {}
            scores = payload.get("scores") or {}
            quality = scores.get("evidence_quality")
            evidence.append(
                {
                    "turn_index": turn.turn_index,
                    "stage": turn.stage,
                    "target_criterion": turn.criterion,
                    "source": source,
                    "weight": evidence_weight,
                    "score": score,
                    "evidence_quality": quality,
                    "strong": score >= 4.0 and (
                        quality is None or float(quality) >= 3.5
                    ),
                }
            )

        if not evidence:
            return (
                {
                    "score": 1.0,
                    "confidence": 0.0,
                    "status": "unanswered_penalty",
                    "evidence_count": 0,
                    "evidence_turns": [],
                    "strong_evidence_turns": [],
                },
                [],
                [
                    {
                        "type": "unanswered_penalty",
                        "competency": competency,
                        "message": f"Ứng viên không cung cấp bằng chứng cho {competency} (câu hỏi bị bỏ qua hoặc không trả lời).",
                    }
                ],
            )

        total_weight = sum(item["weight"] for item in evidence)
        score = sum(item["score"] * item["weight"] for item in evidence) / total_weight
        values = [float(item["score"]) for item in evidence]
        spread = max(values) - min(values)
        deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
        coverage_target = 2.0
        coverage = min(1.0, total_weight / coverage_target)
        agreement = max(0.0, 1.0 - deviation / 2.0)
        quality_values = [
            float(item["evidence_quality"])
            for item in evidence
            if item["evidence_quality"] is not None
        ]
        quality = (
            (statistics.fmean(quality_values) - 1.0) / 4.0
            if quality_values
            else 0.5
        )
        confidence = max(0.0, min(1.0, 0.5 * coverage + 0.3 * agreement + 0.2 * quality))
        direct_count = sum(item["source"] == "direct" for item in evidence)
        status = (
            "sufficient"
            if confidence >= 0.6 and (direct_count >= 2 or competency in {"communication", "role_relevance"})
            else "limited"
        )
        flags: list[dict[str, Any]] = []
        if spread > 1.5:
            flags.append(
                {
                    "type": "score_divergence",
                    "competency": competency,
                    "turns": [item["turn_index"] for item in evidence],
                    "spread": round(spread, 3),
                    "message": "Turn evidence varies substantially for this competency.",
                }
            )
        return (
            {
                "score": round(score, 3),
                "confidence": round(confidence, 3),
                "status": status,
                "evidence_count": len(evidence),
                "direct_evidence_count": direct_count,
                "evidence_turns": [item["turn_index"] for item in evidence],
                "strong_evidence_turns": [
                    item["turn_index"] for item in evidence if item["strong"]
                ],
            },
            evidence,
            flags,
        )

    @staticmethod
    def _consistency_flags(turns: list[Any]) -> list[dict[str, Any]]:
        flags: list[dict[str, Any]] = []
        for turn in turns:
            payload = turn.answer_scores_json or {}
            consistency = (payload.get("scores") or {}).get("consistency")
            if consistency is not None and float(consistency) < 2.5:
                flags.append(
                    {
                        "type": "possible_inconsistency",
                        "turns": [turn.turn_index],
                        "score": round(float(consistency), 3),
                        "message": "This turn has a low predicted consistency score.",
                    }
                )
        return flags

    @classmethod
    def aggregate(cls, session: InterviewSession) -> dict[str, Any]:
        total_expected = max(session.max_questions or 5, len(session.turns))
        all_turns = list(session.turns)

        # Phân loại chi tiết các câu trả lời
        unanswered_turns = []
        non_answer_turns = []
        valid_answered_turns = []

        for turn in all_turns:
            ans = (turn.answer_text or "").strip()
            if not ans:
                unanswered_turns.append(turn.turn_index)
            elif is_non_answer(ans) or (turn.answer_scores_json or {}).get("is_non_answer"):
                non_answer_turns.append(turn.turn_index)
            else:
                valid_answered_turns.append(turn.turn_index)

        # Số lượt bị thiếu nếu phiên bị kết thúc sớm
        missing_count = max(0, total_expected - len(all_turns))

        competencies: dict[str, dict[str, Any]] = {}
        evidence: dict[str, list[dict[str, Any]]] = {}
        flags = cls._consistency_flags(all_turns)

        for competency in BASE_COMPETENCIES:
            result, competency_evidence, competency_flags = cls._aggregate_competency(
                all_turns,
                competency,
            )
            competencies[competency] = result
            evidence[competency] = competency_evidence
            flags.extend(competency_flags)

        # Tính điểm số tổng thể có trừ điểm câu hỏi không trả lời / không biết
        total_weight = sum(OVERALL_WEIGHTS[c] for c in BASE_COMPETENCIES)
        raw_overall = (
            sum(
                float(competencies[c]["score"]) * OVERALL_WEIGHTS[c]
                for c in BASE_COMPETENCIES
            )
            / total_weight
        )

        # Tỷ lệ hoàn thành câu hỏi thực chất (không tính câu bỏ qua/không biết)
        completion_rate = len(valid_answered_turns) / max(1, total_expected)

        # Nếu ứng viên KHÔNG trả lời bất kỳ câu hỏi nào (bỏ qua toàn bộ hoặc kết thúc luôn phiên): Cho 0 điểm
        if len(valid_answered_turns) == 0:
            overall_score = 0.0
            confidence = 0.0
            overall_status = "zero_score"
            for c in BASE_COMPETENCIES:
                competencies[c]["score"] = 0.0
                competencies[c]["confidence"] = 0.0
                competencies[c]["status"] = "zero_score"
            
            flags.append({
                "type": "zero_score_penalty",
                "message": (
                    f"Ứng viên bỏ qua hoặc không trả lời bất kỳ câu hỏi nào (0/{total_expected} câu hoàn thành). "
                    "Hệ thống tự động chấm 0 ĐIỂM toàn diện cho phiên phỏng vấn này."
                ),
                "unanswered_count": len(unanswered_turns) + missing_count,
                "non_answer_count": len(non_answer_turns),
                "valid_count": 0,
                "total_expected": total_expected,
                "completion_rate": 0.0,
            })
        else:
            # Tính điểm số tổng thể có trừ điểm câu hỏi không trả lời / không biết
            total_weight = sum(OVERALL_WEIGHTS[c] for c in BASE_COMPETENCIES)
            raw_overall = (
                sum(
                    float(competencies[c]["score"]) * OVERALL_WEIGHTS[c]
                    for c in BASE_COMPETENCIES
                )
                / total_weight
            )

            # Phạt trực tiếp theo tỷ lệ hoàn thành câu hỏi thực chất
            overall_score = round(raw_overall * completion_rate, 3)

            # Độ tin cậy tính trên cơ sở tỷ lệ câu hỏi thực chất
            base_conf = sum(
                float(competencies[c]["confidence"]) * OVERALL_WEIGHTS[c]
                for c in BASE_COMPETENCIES
            ) / total_weight
            confidence = round(base_conf * completion_rate, 3)

            overall_status = (
                "sufficient"
                if (completion_rate >= 0.8 and confidence >= 0.5)
                else "limited" if completion_rate >= 0.4 else "unanswered_penalty"
            )

            total_penalized_turns = len(unanswered_turns) + len(non_answer_turns) + missing_count
            if total_penalized_turns > 0:
                penalty_details = []
                if non_answer_turns:
                    penalty_details.append(f"{len(non_answer_turns)} câu trả lời không biết/bỏ qua")
                if unanswered_turns:
                    penalty_details.append(f"{len(unanswered_turns)} câu chưa trả lời")
                if missing_count > 0:
                    penalty_details.append(f"{missing_count} câu bị bỏ dở do kết thúc sớm")

                flags.append({
                    "type": "unanswered_penalty",
                    "message": (
                        f"Điểm số tổng cuối cùng đã bị trừ do: {'; '.join(penalty_details)}. "
                        f"Tỷ lệ hoàn thành thực chất: {completion_rate*100:.1f}%."
                    ),
                    "unanswered_count": len(unanswered_turns) + missing_count,
                    "non_answer_count": len(non_answer_turns),
                    "valid_count": len(valid_answered_turns),
                    "total_expected": total_expected,
                    "completion_rate": round(completion_rate, 3),
                })

        overall = {
            "score": round(overall_score, 3),
            "confidence": confidence,
            "status": overall_status,
            "total_expected_questions": total_expected,
            "valid_answered_count": len(valid_answered_turns),
            "non_answer_count": len(non_answer_turns),
            "unanswered_count": len(unanswered_turns) + missing_count,
            "completion_rate": round(completion_rate, 3),
            "covered_competencies": list(BASE_COMPETENCIES),
            "missing_competencies": [
                c for c in BASE_COMPETENCIES if competencies[c]["status"] == "unanswered_penalty"
            ],
        }
        competencies["overall_readiness"] = overall

        return {
            "aggregator_version": AGGREGATOR_VERSION,
            "status": "final" if session.status == "completed" else "provisional",
            "competencies": competencies,
            "evidence": evidence,
            "flags": flags,
        }


class InterviewEvidenceSummaryService:
    @staticmethod
    def get(
        db: Session,
        session_id: str,
    ) -> InterviewEvidenceSummary | None:
        return (
            db.query(InterviewEvidenceSummary)
            .filter(
                InterviewEvidenceSummary.session_id == session_id,
                InterviewEvidenceSummary.aggregator_version == AGGREGATOR_VERSION,
            )
            .first()
        )

    @classmethod
    def create_or_update(
        cls,
        db: Session,
        session: InterviewSession,
        *,
        commit: bool = True,
    ) -> InterviewEvidenceSummary:
        result = InterviewEvidenceAggregator.aggregate(session)
        summary = cls.get(db, session.session_id)
        if summary is None:
            summary = InterviewEvidenceSummary(
                session_id=session.session_id,
                aggregator_version=AGGREGATOR_VERSION,
                status=result["status"],
                competency_scores_json=result["competencies"],
                evidence_json=result["evidence"],
                flags_json=result["flags"],
            )
            db.add(summary)
        else:
            summary.status = result["status"]
            summary.competency_scores_json = result["competencies"]
            summary.evidence_json = result["evidence"]
            summary.flags_json = result["flags"]
            summary.updated_at = datetime.now(timezone.utc)
        if commit:
            db.commit()
            db.refresh(summary)
        else:
            db.flush()
        return summary
