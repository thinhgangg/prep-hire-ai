from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

from sqlalchemy import text
from app.db.session import SessionLocal
from app.services.interview_service import InterviewSessionService
from app.services.synthetic_interview_context_service import SyntheticInterviewContextService
from app.services.humanizer_service import (
    PERSONA_PROMPTS,
    HumanizedAnswerService,
)

logger = logging.getLogger(__name__)


def get_db_session(max_attempts: int = 3):
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            return db
        except Exception as exc:
            last_err = exc
            time.sleep(1.2)
    raise RuntimeError(f"Database connection failed after {max_attempts} attempts: {last_err}")


def print_banner() -> None:
    print("=" * 85)
    print("  TỰ ĐỘNG CHẠY PHỎNG VẤN & SINH DATASET NGƯỜI THẬT (DEBIASING DATASET PIPELINE)")
    print("  Tự sinh câu hỏi thích ứng -> Sinh câu trả lời thô có cảm xúc/ngập ngừng -> Lưu Dataset")
    print("=" * 85)


ROLES_CONFIG = [
    {"role": "backend", "level": "junior", "persona": "nervous_junior"},
    {"role": "frontend", "level": "intern", "persona": "stt_spoken_real"},
    {"role": "data", "level": "junior", "persona": "hesitant_struggling"},
    {"role": "backend", "level": "middle", "persona": "confident_mid"},
    {"role": "devops", "level": "junior", "persona": "conversational_casual"},
    {"role": "mobile", "level": "intern", "persona": "nervous_junior"},
    {"role": "backend", "level": "middle", "persona": "stt_spoken_real"},
]


def run_interview_pipeline(
    sessions_count: int = 3,
    max_questions_per_session: int = 5,
    start_seed: int = 200,
    output_dir_str: Optional[str] = None,
    preferred_provider: Optional[str] = None,
    generate_paired_ai: bool = True,
) -> None:
    print(f"\n[KHỞI ĐỘNG] Chuẩn bị tự động chạy {sessions_count} phiên phỏng vấn...")
    print(f"- Số câu hỏi tối đa mỗi phiên: {max_questions_per_session}")
    print(f"- Sinh cặp đối chiếu AI gốc song song: {'CÓ' if generate_paired_ai else 'KHÔNG'}")

    out_dir = Path(output_dir_str) if output_dir_str else (PROJECT_ROOT / "data" / "augmented")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "debiasing_humanized_dataset.json"
    csv_path = out_dir / "debiasing_humanized_dataset.csv"

    dataset_records: List[Dict[str, Any]] = []
    session_summaries: List[Dict[str, Any]] = []

    for s_idx in range(sessions_count):
        seed = start_seed + s_idx
        role_cfg = ROLES_CONFIG[s_idx % len(ROLES_CONFIG)]
        role = role_cfg["role"]
        level = role_cfg["level"]
        default_persona = role_cfg["persona"]

        print("\n" + "=" * 80)
        print(f"=== BẮT ĐẦU PHIÊN PHỎNG VẤN #{s_idx + 1}/{sessions_count} ===")
        print(f"- Vị trí: {role.upper()} ({level.title()}) | Seed: {seed}")
        print(f"- Phong cách ứng viên chính: {PERSONA_PROMPTS[default_persona]['name']} ({default_persona})")
        print("=" * 80)

        # 1. Sinh ngữ cảnh phỏng vấn (CV và JD)
        context = SyntheticInterviewContextService.generate(seed=seed, role=role, level=level)

        db = get_db_session()
        try:
            # 2. Khởi tạo phiên phỏng vấn mới
            session = InterviewSessionService.create(
                db=db,
                cv_text=context.cv_text,
                jd_text=context.jd_text,
                language="vi",
                max_questions=max_questions_per_session,
                allow_external_ai=True,
                research_consent=False,
                context_source="humanized_synthetic",
            )
            print(f"[HỆ THỐNG] Đã tạo phiên: {session.session_id} (Status: {session.status})")

            # 3. Vòng lặp từng lượt phỏng vấn (Turns)
            turn_counter = 0
            while session.status == "active" and turn_counter < max_questions_per_session:
                current_turn = next(
                    (item for item in session.turns if item.turn_index == session.current_turn_index),
                    None,
                )
                if not current_turn:
                    break

                turn_counter += 1
                q_text = current_turn.question_text
                stage = current_turn.stage
                criterion = current_turn.criterion

                print(f"\n  [Lượt {turn_counter}/{max_questions_per_session}] Stage: {stage} | Tiêu chí: {criterion}")
                print(f"  AI Hỏi: \"{q_text}\"")

                # Chọn persona cho lượt này (có sự biến thiên tự nhiên)
                turn_persona = default_persona
                if turn_counter == 1:
                    turn_persona = "nervous_junior"  # Lượt đầu thường hồi hộp
                elif turn_counter == 3:
                    turn_persona = random.choice(["hesitant_struggling", "stt_spoken_real"])

                # 4. Sinh câu trả lời thô của con người qua Tool
                print(f"  [AI Giả Lập Người] Đang suy nghĩ & trả lời theo phong cách '{turn_persona}'...")
                ans_obj = HumanizedAnswerService.generate_single_persona(
                    question=q_text,
                    context=f"Vai trò: {level} {role}. Trích đoạn CV: {context.cv_text[:300]}",
                    persona=turn_persona,
                    preferred_provider=preferred_provider,
                )

                print(f"  Ứng viên nói: \"{ans_obj.answer_text[:140]}...\"")
                print(
                    f"  -> Chỉ số: {ans_obj.metrics.word_count} từ, "
                    f"{ans_obj.metrics.filler_count} từ đệm, "
                    f"{ans_obj.metrics.pause_count} ngắt (...), "
                    f"Độ thô người thật: {ans_obj.metrics.hesitation_score}/100"
                )

                # Sinh thêm câu trả lời AI gốc (để làm cặp đối chiếu debiasing nếu bật)
                pure_ai_answer = ""
                if generate_paired_ai:
                    pure_ai_prompt = f"Trả lời câu hỏi phỏng vấn sau một cách ngắn gọn, súc tích, chuẩn mực: \"{q_text}\""
                    try:
                        ai_data, _, _, _ = HumanizedAnswerService._execute_llm_json(
                            f'{pure_ai_prompt}\nTrả về JSON: {{"answer": "nội dung"}}',
                            preferred_provider=preferred_provider,
                            operation="pure_ai_baseline",
                        )
                        pure_ai_answer = str(ai_data.get("answer") or "").strip()
                    except Exception:
                        pure_ai_answer = "Giải pháp triển khai theo kiến trúc tiêu chuẩn, đảm bảo tính sẵn sàng cao."

                # 5. Gửi câu trả lời vào hệ thống phỏng vấn để Model 2 chấm điểm & chuyển sang câu tiếp theo
                session = InterviewSessionService.submit_answer(
                    db,
                    session.session_id,
                    ans_obj.answer_text,
                )

                # Lấy điểm số Model 2 vừa chấm cho lượt này
                scored_turn = next(
                    (item for item in session.turns if item.turn_index == current_turn.turn_index),
                    current_turn,
                )
                m2_scores = (scored_turn.answer_scores_json or {}).get("scores", {})

                # 6. Ghi nhận vào tập Dataset huấn luyện
                dataset_record = {
                    "session_id": session.session_id,
                    "turn_index": current_turn.turn_index,
                    "role": role,
                    "level": level,
                    "stage": stage,
                    "criterion": criterion,
                    "question_text": q_text,
                    "humanized_answer": ans_obj.answer_text,
                    "persona": turn_persona,
                    "persona_name": ans_obj.persona_name,
                    "word_count": ans_obj.metrics.word_count,
                    "filler_count": ans_obj.metrics.filler_count,
                    "pause_count": ans_obj.metrics.pause_count,
                    "self_correction_count": ans_obj.metrics.self_correction_count,
                    "hesitation_score": ans_obj.metrics.hesitation_score,
                    "model2_technical_correctness": m2_scores.get("technical_correctness", 0.0),
                    "model2_reasoning_quality": m2_scores.get("reasoning_quality", 0.0),
                    "model2_evidence_quality": m2_scores.get("evidence_quality", 0.0),
                    "model2_expected_signal_coverage": m2_scores.get("expected_signal_coverage", 0.0),
                    "model2_answer_relevance": m2_scores.get("answer_relevance", 0.0),
                    "model2_clarity_conciseness": m2_scores.get("clarity_conciseness", 0.0),
                    "model2_consistency": m2_scores.get("consistency", 0.0),
                    "paired_pure_ai_answer": pure_ai_answer,
                    "generator_provider": ans_obj.provider,
                    "generator_model": ans_obj.model,
                }
                dataset_records.append(dataset_record)

            session_summaries.append(
                {
                    "session_id": session.session_id,
                    "role": role,
                    "level": level,
                    "total_turns": len(session.turns),
                    "status": session.status,
                }
            )
            print(f"\n[KẾT THÚC PHIÊN] Hoàn thành {len(session.turns)} câu hỏi phỏng vấn cho session {session.session_id}.")

        except Exception as exc:
            print(f"[LỖI TRONG PHIÊN] {exc}")
        finally:
            db.close()

    # 7. Xuất Dataset thành JSON & CSV
    print("\n" + "=" * 80)
    print("  ĐANG LƯU DATASET HUẤN LUYỆN (DEBIASING DATASET)")
    print("=" * 80)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_records, f, ensure_ascii=False, indent=2)
    print(f"- [JSON] Đã lưu {len(dataset_records)} mẫu dữ liệu tại: {json_path}")

    if dataset_records:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(dataset_records[0].keys()))
            writer.writeheader()
            writer.writerows(dataset_records)
        print(f"- [CSV]  Đã lưu {len(dataset_records)} mẫu dữ liệu tại: {csv_path}")

    # 8. Báo cáo thống kê phân bổ điểm & đặc trưng
    if dataset_records:
        avg_hesitation = sum(r["hesitation_score"] for r in dataset_records) / len(dataset_records)
        avg_fillers = sum(r["filler_count"] for r in dataset_records) / len(dataset_records)
        avg_tech_score = sum(r["model2_technical_correctness"] for r in dataset_records) / len(dataset_records)
        avg_reasoning = sum(r["model2_reasoning_quality"] for r in dataset_records) / len(dataset_records)

        print("\n--- THỐNG KÊ DATASET MỚI TẠO ---")
        print(f"- Tổng số mẫu câu hỏi - câu trả lời người thật: {len(dataset_records)}")
        print(f"- Điểm mô phỏng người trung bình (Hesitation Score): {avg_hesitation:.1f}/100")
        print(f"- Số từ đệm trung bình/câu: {avg_fillers:.1f}")
        print(f"- Điểm Model 2 kỹ thuật trung bình: {avg_tech_score:.3f} / 5.0")
        print(f"- Điểm Model 2 lập luận trung bình: {avg_reasoning:.3f} / 5.0")
        print("Tập dữ liệu này sẵn sàng để đưa vào huấn luyện lại Model 2!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline tự động chạy phỏng vấn và sinh dataset người thật để giảm thiên vị cho Model 2."
    )
    parser.add_argument("--sessions", type=int, default=3, help="Số lượng phiên phỏng vấn cần chạy (mặc định 3)")
    parser.add_argument("--max-questions", type=int, default=5, help="Số câu hỏi mỗi phiên (mặc định 5)")
    parser.add_argument("--start-seed", type=int, default=200, help="Seed bắt đầu cho hồ sơ (mặc định 200)")
    parser.add_argument("--output-dir", type=str, help="Thư mục xuất file dataset")
    parser.add_argument("--provider", type=str, help="LLM Provider ưu tiên (groq, cloudflare...)")
    parser.add_argument("--no-paired-ai", action="store_true", help="Không sinh câu trả lời AI gốc song song")

    args = parser.parse_args()
    print_banner()

    run_interview_pipeline(
        sessions_count=args.sessions,
        max_questions_per_session=args.max_questions,
        start_seed=args.start_seed,
        output_dir_str=args.output_dir,
        preferred_provider=args.provider,
        generate_paired_ai=not args.no_paired_ai,
    )


if __name__ == "__main__":
    main()

