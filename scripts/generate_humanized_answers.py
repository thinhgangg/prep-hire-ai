from __future__ import annotations

import argparse
import csv
import json
import os
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
from app.models.interview_turn import InterviewTurn
from app.services.humanizer_service import (
    PERSONA_PROMPTS,
    HumanizedAnswer,
    HumanizedAnswerService,
    MultiPersonaResult,
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


def print_banner() -> None:
    print("=" * 80)
    print("  HUMANIZED INTERVIEW ANSWER GENERATOR TOOL - PREP-HIRE AI")
    print("  Mô phỏng câu trả lời phỏng vấn người thật: Ngập ngừng, Cảm xúc, Văn nói tự nhiên")
    print("=" * 80)


def score_answer_model2(
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


def print_score_comparison(ai_scores: Dict[str, float], human_scores: Dict[str, float]) -> None:
    if not ai_scores and not human_scores:
        return
    print("\n  [SO SÁNH ĐIỂM SỐ MÔ HÌNH 2 (CANDIDATE ANSWER SCORER)]")
    print("  {:<30} | {:<12} | {:<12} | {:<10}".format("Tiêu chí đánh giá", "AI Gốc (1-5)", "Người Thật", "Chênh lệch"))
    print("  " + "-" * 72)
    for k in [
        "technical_correctness",
        "reasoning_quality",
        "evidence_quality",
        "expected_signal_coverage",
        "answer_relevance",
        "clarity_conciseness",
        "consistency",
    ]:
        s_ai = ai_scores.get(k, 0.0)
        s_hu = human_scores.get(k, 0.0)
        diff = s_hu - s_ai
        diff_str = f"{diff:+.3f}"
        print(f"  {k:<30} | {s_ai:<12.3f} | {s_hu:<12.3f} | {diff_str:<10}")
    print("  " + "-" * 72)


def print_persona_result(persona_key: str, ans: HumanizedAnswer) -> None:
    print(f"\n[PHONG CÁCH: {ans.persona_name}] ({persona_key})")
    print(f"- Nhà cung cấp AI: {ans.provider} ({ans.model}) | Độ trễ: {ans.latency_ms}ms")
    print(
        f"- Chỉ số văn nói: {ans.metrics.word_count} từ | "
        f"{ans.metrics.filler_count} từ đệm | "
        f"{ans.metrics.pause_count} khoảng lặng (...) | "
        f"{ans.metrics.self_correction_count} lần tự sửa lời | "
        f"Điểm mô phỏng người: {ans.metrics.hesitation_score}/100"
    )
    print("-" * 80)
    print(f"\"{ans.answer_text}\"")
    print("-" * 80)


def run_interactive_mode(provider: Optional[str] = None, compare_scores: bool = False) -> None:
    print("\n--- CHẾ ĐỘ TƯƠNG TÁC DÒNG LỆNH (INTERACTIVE PLAYGROUND) ---")
    print("Nhập 'q' hoặc 'exit' để thoát.\n")

    available_personas = list(PERSONA_PROMPTS.keys())

    while True:
        try:
            print("\n" + "=" * 60)
            question = input(">> Nhập câu hỏi phỏng vấn: ").strip()
            if not question:
                continue
            if question.lower() in ("q", "exit", "quit"):
                print("Đã thoát chế độ tương tác.")
                break

            print("\nChọn tùy chọn:")
            print("1. Sinh tất cả 5 phong cách người thật cùng lúc")
            print("2. Chuyển đổi một câu trả lời AI khô cứng thành văn nói người (có so sánh điểm)")
            print("3. Chọn 1 phong cách cụ thể")
            choice = input("Lựa chọn của bạn (1/2/3, mặc định 1): ").strip() or "1"

            if choice == "2":
                ai_answer = input(">> Dán câu trả lời AI gốc (văn mẫu): ").strip()
                if not ai_answer:
                    print("Bỏ qua do không có nội dung AI.")
                    continue
                print("\nCác phong cách khả dụng:")
                for i, p in enumerate(available_personas, 1):
                    print(f"  {i}. {p} - {PERSONA_PROMPTS[p]['name']}")
                p_idx = input("Chọn số phong cách (1-5, mặc định 1): ").strip()
                selected_p = (
                    available_personas[int(p_idx) - 1]
                    if p_idx.isdigit() and 1 <= int(p_idx) <= len(available_personas)
                    else "nervous_junior"
                )
                print(f"\n[AI] Đang chuyển đổi sang phong cách '{selected_p}'...")
                res = HumanizedAnswerService.humanize_existing_answer(
                    question=question,
                    raw_ai_answer=ai_answer,
                    persona=selected_p,
                    preferred_provider=provider,
                )
                print_persona_result(selected_p, res)

                if compare_scores:
                    print("\n[Mô hình 2] Đang chấm điểm so sánh...")
                    ai_scores = score_answer_model2(question, ai_answer)
                    human_scores = score_answer_model2(question, res.answer_text)
                    print_score_comparison(ai_scores, human_scores)

            elif choice == "3":
                print("\nCác phong cách khả dụng:")
                for i, p in enumerate(available_personas, 1):
                    print(f"  {i}. {p} - {PERSONA_PROMPTS[p]['name']}")
                p_idx = input("Chọn số phong cách (1-5, mặc định 1): ").strip()
                selected_p = (
                    available_personas[int(p_idx) - 1]
                    if p_idx.isdigit() and 1 <= int(p_idx) <= len(available_personas)
                    else "nervous_junior"
                )
                print(f"\n[AI] Đang sinh câu trả lời phong cách '{selected_p}'...")
                res = HumanizedAnswerService.generate_single_persona(
                    question=question,
                    persona=selected_p,
                    preferred_provider=provider,
                )
                print_persona_result(selected_p, res)

            else:
                print("\n[AI] Đang sinh cùng lúc 5 phong cách người thật...")
                multi_res = HumanizedAnswerService.generate_all_personas(
                    question=question,
                    preferred_provider=provider,
                )
                for p_key, ans_obj in multi_res.answers.items():
                    print_persona_result(p_key, ans_obj)

        except KeyboardInterrupt:
            print("\nĐã hủy.")
            break
        except Exception as exc:
            print(f"[LỖI] {exc}")


def run_db_augmentation(
    limit: int = 10,
    personas: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    provider: Optional[str] = None,
    compare_scores: bool = False,
) -> None:
    print(f"\n--- ĐANG QUÉT CÂU HỎI THẬT TỪ DATABASE (Tối đa: {limit} câu) ---")
    db = get_db_session()
    try:
        turns = (
            db.query(InterviewTurn)
            .filter(
                InterviewTurn.question_text.isnot(None),
                InterviewTurn.question_text != "",
            )
            .order_by(InterviewTurn.turn_id.desc())
            .limit(limit * 2)
            .all()
        )

        seen_questions = set()
        selected_turns: List[InterviewTurn] = []
        for t in turns:
            q_clean = t.question_text.strip()
            if q_clean not in seen_questions:
                seen_questions.add(q_clean)
                selected_turns.append(t)
            if len(selected_turns) >= limit:
                break

        print(f"Tìm thấy {len(selected_turns)} câu hỏi phỏng vấn thực tế từ database.")
        if not selected_turns:
            print("Không tìm thấy câu hỏi trong database.")
            return

        target_personas = personas or ["nervous_junior", "confident_mid", "stt_spoken_real"]
        print(f"Sẽ sinh {len(target_personas)} phong cách/câu: {', '.join(target_personas)}")

        all_records: List[Dict[str, Any]] = []

        for idx, turn in enumerate(selected_turns, 1):
            print(f"\n[{idx}/{len(selected_turns)}] Đang xử lý câu hỏi Turn #{turn.turn_id} ({turn.stage}):")
            print(f"    Q: {turn.question_text[:110]}...")

            context = f"Stage: {turn.stage}, Criterion: {turn.criterion}"
            if turn.expected_signal:
                context += f", Expected Signal: {turn.expected_signal}"

            try:
                res = HumanizedAnswerService.generate_all_personas(
                    question=turn.question_text,
                    context=context,
                    personas=target_personas,
                    preferred_provider=provider,
                )

                for p_key, ans_obj in res.answers.items():
                    record = {
                        "source_turn_id": turn.turn_id,
                        "session_id": turn.session_id,
                        "stage": turn.stage,
                        "criterion": turn.criterion,
                        "question_text": turn.question_text,
                        "persona": p_key,
                        "persona_name": ans_obj.persona_name,
                        "humanized_answer": ans_obj.answer_text,
                        "word_count": ans_obj.metrics.word_count,
                        "filler_count": ans_obj.metrics.filler_count,
                        "pause_count": ans_obj.metrics.pause_count,
                        "self_correction_count": ans_obj.metrics.self_correction_count,
                        "hesitation_score": ans_obj.metrics.hesitation_score,
                        "generator_provider": ans_obj.provider,
                        "generator_model": ans_obj.model,
                        "latency_ms": ans_obj.latency_ms,
                        "original_real_user_answer": turn.answer_text,
                    }
                    if compare_scores:
                        m2_scores = score_answer_model2(turn.question_text, ans_obj.answer_text, turn.stage, turn.criterion)
                        record["model2_scores"] = m2_scores

                    all_records.append(record)

                print(f"    -> Đã sinh thành công {len(res.answers)} biến thể văn nói.")

            except Exception as exc:
                print(f"    -> [LỖI khi sinh]: {exc}")

        # Resolve output path
        if not output_path:
            output_dir = PROJECT_ROOT / "data" / "augmented"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / "humanized_interview_answers.json")

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if out_p.suffix.lower() == ".csv":
            if all_records:
                with open(out_p, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
                    writer.writeheader()
                    writer.writerows(all_records)
                print(f"\n[XONG] Đã lưu {len(all_records)} mẫu dữ liệu vào CSV: {out_p}")
        else:
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(all_records, f, ensure_ascii=False, indent=2)
            print(f"\n[XONG] Đã lưu {len(all_records)} mẫu dữ liệu vào JSON: {out_p}")

    finally:
        db.close()


def run_single_question(
    question: str,
    persona: Optional[str] = None,
    context: Optional[str] = None,
    output_path: Optional[str] = None,
    provider: Optional[str] = None,
) -> None:
    print(f"\n--- SINH CÂU TRẢ LỜI CHO CÂU HỎI: \"{question}\" ---")
    if persona and persona in PERSONA_PROMPTS:
        res = HumanizedAnswerService.generate_single_persona(
            question=question,
            context=context,
            persona=persona,
            preferred_provider=provider,
        )
        print_persona_result(persona, res)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(res.as_dict(), f, ensure_ascii=False, indent=2)
            print(f"\nĐã lưu kết quả vào: {output_path}")
    else:
        multi_res = HumanizedAnswerService.generate_all_personas(
            question=question,
            context=context,
            preferred_provider=provider,
        )
        for p_key, ans_obj in multi_res.answers.items():
            print_persona_result(p_key, ans_obj)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(multi_res.as_dict(), f, ensure_ascii=False, indent=2)
            print(f"\nĐã lưu kết quả vào: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tool sinh câu trả lời phỏng vấn AI có cảm xúc, ngập ngừng, tự nhiên như người thật."
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "db", "question", "rewrite"],
        default="interactive",
        help="Chế độ hoạt động: interactive (dòng lệnh trực tiếp), db (lấy từ database), question (câu hỏi đơn), rewrite (viết lại câu trả lời AI)",
    )
    parser.add_argument("--question", type=str, help="Nội dung câu hỏi phỏng vấn (dùng cho mode=question hoặc rewrite)")
    parser.add_argument("--ai-answer", type=str, help="Nội dung câu trả lời AI gốc (dùng cho mode=rewrite)")
    parser.add_argument("--context", type=str, help="Bối cảnh CV hoặc JD bổ trợ")
    parser.add_argument(
        "--persona",
        type=str,
        default="all",
        help="Phong cách ứng viên: nervous_junior, confident_mid, hesitant_struggling, conversational_casual, stt_spoken_real, hoặc all",
    )
    parser.add_argument("--limit", type=int, default=5, help="Số lượng câu hỏi lấy từ DB (mặc định 5)")
    parser.add_argument("--output", type=str, help="Đường dẫn file lưu kết quả (.json hoặc .csv)")
    parser.add_argument("--provider", type=str, help="LLM provider ưu tiên (groq, cloudflare, inception, opentyphoon...)")
    parser.add_argument("--compare-scores", action="store_true", help="Chấm điểm so sánh bằng Mô hình 2 (Candidate Answer Scorer)")

    args = parser.parse_args()
    print_banner()

    personas_list = None
    if args.persona and args.persona != "all":
        personas_list = [p.strip() for p in args.persona.split(",") if p.strip() in PERSONA_PROMPTS]

    if args.mode == "interactive":
        run_interactive_mode(provider=args.provider, compare_scores=args.compare_scores)

    elif args.mode == "db":
        run_db_augmentation(
            limit=args.limit,
            personas=personas_list,
            output_path=args.output,
            provider=args.provider,
            compare_scores=args.compare_scores,
        )

    elif args.mode == "question":
        if not args.question:
            print("[LỖI] Chế độ 'question' yêu cầu tham số --question \"...\"")
            sys.exit(1)
        run_single_question(
            question=args.question,
            persona=args.persona if args.persona != "all" else None,
            context=args.context,
            output_path=args.output,
            provider=args.provider,
        )

    elif args.mode == "rewrite":
        if not args.question or not args.ai_answer:
            print("[LỖI] Chế độ 'rewrite' yêu cầu cả --question \"...\" và --ai-answer \"...\"")
            sys.exit(1)
        selected_persona = args.persona if args.persona in PERSONA_PROMPTS else "nervous_junior"
        print(f"\n[AI] Đang chuyển đổi câu trả lời sang phong cách '{selected_persona}'...")
        res = HumanizedAnswerService.humanize_existing_answer(
            question=args.question,
            raw_ai_answer=args.ai_answer,
            persona=selected_persona,
            preferred_provider=args.provider,
        )
        print_persona_result(selected_persona, res)

        if args.compare_scores:
            print("\n[Mô hình 2] Đang chấm điểm so sánh...")
            ai_scores = score_answer_model2(args.question, args.ai_answer)
            human_scores = score_answer_model2(args.question, res.answer_text)
            print_score_comparison(ai_scores, human_scores)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(res.as_dict(), f, ensure_ascii=False, indent=2)
            print(f"\nĐã lưu kết quả vào: {args.output}")


if __name__ == "__main__":
    main()
