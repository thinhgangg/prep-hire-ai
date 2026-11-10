from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

from sqlalchemy import text
from app.db.session import SessionLocal
from app.models.interview_turn import InterviewTurn
from app.services.humanizer_service import (
    PERSONA_PROMPTS,
    HumanizedAnswerService,
)


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


def score_with_model2(
    question: str,
    answer: str,
    stage: str = "technical",
    criterion: str = "technical_knowledge",
) -> Dict[str, float]:
    try:
        from app.services.interview_model_service import InterviewModelService
        scores, _ = InterviewModelService.score_answer(
            context_text=f"Stage: {stage}, Criterion: {criterion}",
            question=question,
            answer=answer,
            stage=stage,
            target_criterion=criterion,
        )
        return scores
    except Exception:
        return {}


def fetch_candidate_questions(limit: int = 200, exclude_questions: Optional[set] = None) -> List[Dict[str, str]]:
    db = get_db_session()
    questions: List[Dict[str, str]] = []
    seen = set(exclude_questions) if exclude_questions else set()
    try:
        # 1. Real questions from interview_turns
        turns = (
            db.query(InterviewTurn.question_text, InterviewTurn.stage, InterviewTurn.criterion)
            .filter(InterviewTurn.question_text.isnot(None), InterviewTurn.question_text != "")
            .order_by(InterviewTurn.turn_id.desc())
            .all()
        )
        for q, stage, crit in turns:
            q_clean = (q or "").strip()
            if q_clean and len(q_clean) > 20 and q_clean not in seen:
                seen.add(q_clean)
                questions.append({
                    "question_text": q_clean,
                    "stage": stage or "technical",
                    "criterion": crit or "technical_knowledge",
                    "source": "interview_turns"
                })
            if len(questions) >= limit:
                break

        # 2. Benchmark questions from question_benchmark_outputs
        if len(questions) < limit:
            remaining = limit - len(questions)
            rows = db.execute(
                text("""
                    SELECT question_text, criterion 
                    FROM question_benchmark_outputs 
                    WHERE question_text IS NOT NULL AND length(question_text) > 20
                    ORDER BY RANDOM() LIMIT :lim
                """),
                {"lim": remaining * 3}
            ).fetchall()
            for r in rows:
                q_clean = (r[0] or "").strip()
                if q_clean and len(q_clean) > 20 and q_clean not in seen:
                    seen.add(q_clean)
                    questions.append({
                        "question_text": q_clean,
                        "stage": "technical",
                        "criterion": r[1] or "technical_knowledge",
                        "source": "benchmark"
                    })
                if len(questions) >= limit:
                    break

    finally:
        db.close()

    return questions


def generate_paired_sample(
    question: str,
    persona_key: str,
    stage: str,
    criterion: str,
    sample_index: int,
) -> Dict[str, Any]:
    persona_info = PERSONA_PROMPTS.get(persona_key, PERSONA_PROMPTS["nervous_junior"])

    prompt = f"""Bạn là một chuyên gia ngôn ngữ học phỏng vấn tiếng Việt.
Nhiệm vụ: Với câu hỏi phỏng vấn dưới đây, hãy sinh ra ĐỒNG THỜI 2 câu trả lời để tạo cặp đối chiếu cân bằng dữ liệu:

CÂU HỎI PHỎNG VẤN:
"{question}"

1. "humanized_answer": Câu trả lời THEO GIỌNG NÓI THẬT CỦA CON NGƯỜI.
- Phong cách: {persona_info['name']}
- Đặc điểm: {persona_info['description']}
- Hướng dẫn:
{persona_info['guidelines']}
- YÊU CẦU BẮT BUỘC: 100% văn nói tiếng Việt, có từ đệm ("dạ...", "ừm...", "thì..."), khoảng ngắt "...", tự sửa lời ("à không ý em là..."), TUYỆT ĐỐI KHÔNG dùng gạch đầu dòng, danh sách 1 2 3 hay văn phong AI.

2. "pure_ai_answer": Câu trả lời THEO PHONG CÁCH AI CHUẨN MỰC (đối chiếu).
- Văn phong chuẩn mực, súc tích, logic, cấu trúc trực diện.

Trả về đúng định dạng JSON:
{{
  "humanized_answer": "nội dung câu trả lời nói miệng tự nhiên",
  "pure_ai_answer": "nội dung câu trả lời chuẩn AI"
}}
"""
    started = time.perf_counter()
    data, provider, model, latency_ms = HumanizedAnswerService._execute_llm_json(
        prompt,
        operation=f"batch_paired_{persona_key}",
        temperature=0.75,
    )
    total_latency_ms = round((time.perf_counter() - started) * 1000)

    humanized_text = str(data.get("humanized_answer") or "").strip()
    pure_ai_text = str(data.get("pure_ai_answer") or "").strip()

    if not humanized_text:
        humanized_text = "Dạ... ừm... về câu hỏi này thì em nghĩ là... lúc trước em có làm qua rồi ạ."
    if not pure_ai_text:
        pure_ai_text = "Giải pháp được thực hiện theo tiêu chuẩn kỹ thuật phù hợp."

    metrics = HumanizedAnswerService.calculate_metrics(humanized_text)

    # Score with Model 2
    m2_scores = score_with_model2(question, humanized_text, stage, criterion)

    record = {
        "sample_id": f"humanized_{int(time.time())}_{sample_index}",
        "turn_index": sample_index,
        "source": "augmented_paired",
        "stage": stage,
        "criterion": criterion,
        "question_text": question,
        "humanized_answer": humanized_text,
        "persona": persona_key,
        "persona_name": persona_info["name"],
        "word_count": metrics.word_count,
        "filler_count": metrics.filler_count,
        "pause_count": metrics.pause_count,
        "self_correction_count": metrics.self_correction_count,
        "hesitation_score": metrics.hesitation_score,
        "model2_technical_correctness": m2_scores.get("technical_correctness", 0.0),
        "model2_reasoning_quality": m2_scores.get("reasoning_quality", 0.0),
        "model2_evidence_quality": m2_scores.get("evidence_quality", 0.0),
        "model2_expected_signal_coverage": m2_scores.get("expected_signal_coverage", 0.0),
        "model2_answer_relevance": m2_scores.get("answer_relevance", 0.0),
        "model2_clarity_conciseness": m2_scores.get("clarity_conciseness", 0.0),
        "model2_consistency": m2_scores.get("consistency", 0.0),
        "paired_pure_ai_answer": pure_ai_text,
        "generator_provider": provider,
        "generator_model": model,
        "latency_ms": total_latency_ms,
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch sinh số lượng lớn câu trả lời thô người thật để cân bằng Dataset chống thiên vị."
    )
    parser.add_argument("--count", type=int, default=150, help="Số lượng mẫu câu hỏi-trả lời cần sinh (mặc định 150)")
    parser.add_argument("--workers", type=int, default=5, help="Số luồng chạy song song (mặc định 5)")
    parser.add_argument("--append", action="store_true", default=True, help="Gộp thêm vào dataset hiện có")
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè file dataset mới hoàn toàn")
    args = parser.parse_args()

    print("=" * 85)
    print("  BATCH GENERATOR (PAIRED FAST MODE): SINH DỮ LIỆU THÔ CÂN BẰNG TẬP HUẤN LUYỆN")
    print(f"  Mục tiêu: Sinh {args.count} câu thô kèm câu AI đối chiếu qua {args.workers} luồng song song")
    print("=" * 85)

    out_dir = PROJECT_ROOT / "data" / "augmented"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "debiasing_humanized_dataset.json"
    csv_path = out_dir / "debiasing_humanized_dataset.csv"

    existing_records: List[Dict[str, Any]] = []
    if not args.overwrite and json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                existing_records = json.load(f)
            print(f"- Đã nạp {len(existing_records)} mẫu hiện có từ: {json_path}")
        except Exception:
            existing_records = []

    print("\n[1/3] Đang lấy danh sách câu hỏi phỏng vấn đa dạng từ Database...")
    exclude_q = set(r.get("question_text", "").strip() for r in existing_records)
    candidate_questions = fetch_candidate_questions(limit=args.count, exclude_questions=exclude_q)
    print(f"- Đã lấy được {len(candidate_questions)} câu hỏi phỏng vấn mới chưa từng xuất hiện.")

    personas_pool = ["nervous_junior", "confident_mid", "hesitant_struggling", "conversational_casual", "stt_spoken_real"]

    print(f"\n[2/3] Bắt đầu sinh {len(candidate_questions)} mẫu dữ liệu thô song song (5 luồng)...")
    started_total = time.perf_counter()
    new_records: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for idx, item in enumerate(candidate_questions, 1):
            assigned_persona = personas_pool[(idx - 1) % len(personas_pool)]
            fut = executor.submit(
                generate_paired_sample,
                item["question_text"],
                assigned_persona,
                item["stage"],
                item["criterion"],
                len(existing_records) + idx,
            )
            futures[fut] = (idx, item, assigned_persona)

        done_count = 0
        for fut in as_completed(futures):
            idx, item, p = futures[fut]
            try:
                rec = fut.result()
                new_records.append(rec)
                done_count += 1
                q_snippet = item["question_text"][:50]
                print(f"  [{done_count}/{len(candidate_questions)}] #{idx} ({p}): \"{q_snippet}...\" -> {rec['hesitation_score']}/100 thô ({rec['latency_ms']}ms)", flush=True)
            except Exception as exc:
                print(f"  [{done_count}/{len(candidate_questions)}] LỖI Turn #{idx}: {exc}", flush=True)

    total_time = round(time.perf_counter() - started_total, 1)
    print(f"\n-> Hoàn thành sinh {len(new_records)} mẫu mới trong {total_time}s (~{len(new_records)/max(1, total_time):.1f} mẫu/s).")

    # Combine
    combined_records = existing_records + new_records

    print(f"\n[3/3] Đang lưu tổng cộng {len(combined_records)} mẫu vào file Dataset...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined_records, f, ensure_ascii=False, indent=2)
    print(f"- [JSON] Đã lưu {len(combined_records)} mẫu tại: {json_path}")

    import pandas as pd
    df_out = pd.DataFrame(combined_records)
    df_out.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"- [CSV]  Đã lưu {len(combined_records)} mẫu tại: {csv_path}")

    # Summary Stats
    avg_hesitation = sum(r.get("hesitation_score", 0) for r in combined_records) / len(combined_records)
    avg_fillers = sum(r.get("filler_count", 0) for r in combined_records) / len(combined_records)
    print("\n" + "=" * 80)
    print("  BÁO CÁO TỔNG QUAN TẬP DỮ LIỆU ĐÃ CÂN BẰNG")
    print("=" * 80)
    print(f"- Tổng số mẫu dữ liệu thô người thật: {len(combined_records)}")
    print(f"- Điểm mô phỏng người (Hesitation Score) trung bình: {avg_hesitation:.1f}/100")
    print(f"- Mật độ từ đệm (fillers) trung bình: {avg_fillers:.1f} từ/câu")
    print("- Phân bổ phong cách:")
    for p in personas_pool:
        p_count = sum(1 for r in combined_records if r.get("persona") == p)
        print(f"    * {p:<25}: {p_count} mẫu ({p_count/len(combined_records)*100:.1f}%)")
    print("=" * 80)
    print("Tập dữ liệu đã sẵn sàng để tiến hành huấn luyện (Retrain) Model 2!")


if __name__ == "__main__":
    main()
