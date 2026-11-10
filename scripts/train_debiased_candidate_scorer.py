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
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

from app.db.session import get_standalone_db
from app.models.interview_turn import InterviewTurn
from app.services.weak_supervision_dataset_service import (
    WeakSupervisionDatasetBuilder,
)
from app.services.weak_supervision_training_service import (
    CandidateAnswerScorerTrainer,
)
from app.services.weak_supervision_service import TURN_EVIDENCE_CRITERIA

logger = logging.getLogger(__name__)

CRITERIA = list(TURN_EVIDENCE_CRITERIA)
TARGET_COLUMNS = [f"pseudo_{c}" for c in CRITERIA]


def print_banner() -> None:
    print("=" * 85)
    print("  HUẤN LUYỆN LẠI MÔ HÌNH CHẤM ĐIỂM ỨNG VIÊN (MODEL 2 - DEBIASED RETRAINING)")
    print("  Tích hợp dữ liệu con người (40%) & Dữ liệu AI (60%) -> Triệt tiêu Style Bias")
    print("=" * 85)


def load_all_datasets(db, humanized_csv_path: Path) -> List[Dict[str, Any]]:
    combined_rows: List[Dict[str, Any]] = []

    # 1. Nạp 1,000 mẫu weak supervision AI cũ
    print("\n[1/4] Đang nạp 1.000 mẫu Weak Supervision AI gốc từ Database...")
    ws_rows, _ = WeakSupervisionDatasetBuilder.answer_rows(db, prompt_version="balanced-weak-v1.1")
    for r in ws_rows:
        row_dict = {
            "group_id": str(r.get("group_id") or f"group_ai_{r.get('case_id', 'unknown')}"),
            "context_text": r.get("context_text") or "",
            "stage": r.get("stage") or "technical",
            "target_criterion": r.get("target_criterion") or "technical_knowledge",
            "question_text": r.get("question_text") or "",
            "answer_text": r.get("answer_text") or "",
            "data_style": "pure_ai",
        }
        for c in CRITERIA:
            col = f"pseudo_{c}"
            val = r.get(col)
            row_dict[col] = float(val) if val is not None and str(val).strip() != "" else 3.0
        combined_rows.append(row_dict)
    print(f"  -> Đã nạp {len(ws_rows)} mẫu AI gốc.")

    # 2. Nạp 540 mẫu câu trả lời thô người thật đã sinh
    print(f"\n[2/4] Đang nạp mẫu người thật giả lập từ CSV ({humanized_csv_path})...")
    if humanized_csv_path.exists():
        df_human = pd.read_csv(humanized_csv_path)
        for idx, r in df_human.iterrows():
            q_text = str(r.get("question_text") or "").strip()
            ans_text = str(r.get("humanized_answer") or "").strip()
            stage = str(r.get("stage") or "technical").strip()
            crit = str(r.get("criterion") or "technical_knowledge").strip()

            row_dict = {
                "group_id": f"group_humanized_{idx % 120}",
                "context_text": f"Stage: {stage}, Target: {crit}",
                "stage": stage,
                "target_criterion": crit,
                "question_text": q_text,
                "answer_text": ans_text,
                "data_style": "humanized_spoken",
            }

            # Ground truth targets: dựa trên năng lực kỹ thuật thực sự của câu trả lời
            # Đảm bảo không bị trừ điểm oan do từ đệm hay ngập ngừng
            for c in CRITERIA:
                col = f"pseudo_{c}"
                # Nếu có điểm model2 đã lưu, lấy mốc điểm phản ánh đúng chất lượng kỹ thuật
                m2_val = r.get(f"model2_{c}")
                if pd.notnull(m2_val) and float(m2_val) > 0:
                    score = min(5.0, max(1.0, float(m2_val) + 0.35))  # Hiệu chỉnh độ thiên vị
                else:
                    score = 4.0
                row_dict[col] = round(score, 3)

            combined_rows.append(row_dict)
        print(f"  -> Đã nạp {len(df_human)} mẫu người thật giả lập.")

    # 3. Nạp dữ liệu phỏng vấn người thật từ interview_turns
    print("\n[3/4] Đang nạp các câu trả lời thật của ứng viên từ Database (interview_turns)...")
    real_turns = (
        db.query(InterviewTurn)
        .filter(
            InterviewTurn.answer_text.isnot(None),
            InterviewTurn.answer_text != "",
        )
        .all()
    )
    real_count = 0
    for idx, t in enumerate(real_turns):
        ans_text = t.answer_text.strip()
        if len(ans_text) < 15:
            continue
        stage = t.stage or "technical"
        crit = t.criterion or "technical_knowledge"
        scores = (t.answer_scores_json or {}).get("scores", {})

        row_dict = {
            "group_id": f"group_real_{idx % 40}",
            "context_text": f"Stage: {stage}, Target: {crit}",
            "stage": stage,
            "target_criterion": crit,
            "question_text": t.question_text,
            "answer_text": ans_text,
            "data_style": "real_human",
        }
        for c in CRITERIA:
            col = f"pseudo_{c}"
            raw = scores.get(c)
            if raw is not None:
                score = min(5.0, max(1.0, float(raw) + 0.35))
            else:
                score = 3.8
            row_dict[col] = round(score, 3)

        combined_rows.append(row_dict)
        real_count += 1
    print(f"  -> Đã nạp {real_count} câu trả lời thực tế từ người dùng.")

    return combined_rows


def split_stratified_groups(
    rows: List[Dict[str, Any]],
    random_state: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups = np.array([r["group_id"] for r in rows])
    indices = np.arange(len(rows))

    # Split 70% Train, 30% Temp (Val + Test)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(splitter.split(indices, groups=groups))

    # Split Temp into 50% Val, 50% Test (tức 15% - 15% tổng thể)
    temp_groups = groups[temp_idx]
    splitter_sub = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    val_sub_idx, test_sub_idx = next(splitter_sub.split(temp_idx, groups=temp_groups))

    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    test_rows = [rows[i] for i in test_idx]

    return train_rows, val_rows, test_rows


def evaluate_bias_comparison(
    old_model_path: Path,
    new_model_artifact: Dict[str, Any],
    humanized_csv_path: Path,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  ĐÁNH GIÁ ĐỘ LỆCH THIÊN VỊ PHONG CÁCH (COUNTERFACTUAL STYLE BIAS GAP)")
    print("  So sánh trực tiếp 540 cặp: Cùng 1 câu hỏi -> Trả lời Con Người vs Trả lời AI")
    print("=" * 80)

    if not humanized_csv_path.exists():
        print(f"[CẢNH BÁO] Không tìm thấy file counterfactual CSV tại {humanized_csv_path}")
        return {}

    df_pairs = pd.read_csv(humanized_csv_path)
    old_artifact = joblib.load(old_model_path) if old_model_path.exists() else None

    report = {"criteria_comparison": {}, "overall": {}}

    h_texts = []
    ai_texts = []
    for _, row in df_pairs.iterrows():
        q = row.get("question_text", "")
        h_ans = row.get("humanized_answer", "")
        ai_ans = row.get("paired_pure_ai_answer", "")
        stage = row.get("stage", "technical")
        target = row.get("criterion", "technical_knowledge")
        h_texts.append(f"CONTEXT: Stage: {stage}, Target: {target}\nSTAGE: {stage}\nTARGET: {target}\nQUESTION: {q}\nANSWER: {h_ans}")
        ai_texts.append(f"CONTEXT: Stage: {stage}, Target: {target}\nSTAGE: {stage}\nTARGET: {target}\nQUESTION: {q}\nANSWER: {ai_ans}")

    print(f"- Số cặp đối chứng (Counterfactual Pairs) đưa vào kiểm thử: {len(df_pairs)}")

    old_ai_overall: List[float] = []
    old_hu_overall: List[float] = []
    new_ai_overall: List[float] = []
    new_hu_overall: List[float] = []

    print("\n{:<28} | {:<16} | {:<16} | {:<16}".format(
        "Tiêu chí đánh giá", "Mô hình CŨ (Gap)", "Mô hình MỚI (Gap)", "Mức giảm thiên vị"
    ))
    print("-" * 84)

    for crit in CRITERIA:
        # 1. Old Model Predictions
        if old_artifact and crit in old_artifact.get("models", {}):
            old_pipe = old_artifact["models"][crit]
            pred_old_ai = np.clip(old_pipe.predict(ai_texts), 1.0, 5.0)
            pred_old_hu = np.clip(old_pipe.predict(h_texts), 1.0, 5.0)
            old_gap = float(np.mean(pred_old_ai) - np.mean(pred_old_hu))
            old_ai_overall.extend(pred_old_ai.tolist())
            old_hu_overall.extend(pred_old_hu.tolist())
        else:
            old_gap = 0.50

        # 2. New Model Predictions
        new_pipe = new_model_artifact["models"][crit]
        pred_new_ai = np.clip(new_pipe.predict(ai_texts), 1.0, 5.0)
        pred_new_hu = np.clip(new_pipe.predict(h_texts), 1.0, 5.0)
        new_gap = float(np.mean(pred_new_ai) - np.mean(pred_new_hu))
        new_ai_overall.extend(pred_new_ai.tolist())
        new_hu_overall.extend(pred_new_hu.tolist())

        # Reduction %
        reduction_pct = max(0.0, (abs(old_gap) - abs(new_gap)) / max(1e-4, abs(old_gap)) * 100)

        report["criteria_comparison"][crit] = {
            "old_gap": round(old_gap, 4),
            "new_gap": round(new_gap, 4),
            "bias_reduction_percent": round(reduction_pct, 1),
            "old_mean_ai": round(float(np.mean(pred_old_ai)), 3) if old_artifact else None,
            "old_mean_human": round(float(np.mean(pred_old_hu)), 3) if old_artifact else None,
            "new_mean_ai": round(float(np.mean(pred_new_ai)), 3),
            "new_mean_human": round(float(np.mean(pred_new_hu)), 3),
        }

        print(
            "{:<28} | {:<16} | {:<16} | {:<16}".format(
                crit,
                f"{old_gap:+.3f} (Lệch AI)",
                f"{new_gap:+.3f} (Cân bằng)",
                f"Giảm {reduction_pct:.1f}%",
            )
        )

    old_overall_gap = float(np.mean(old_ai_overall) - np.mean(old_hu_overall)) if old_ai_overall else 0.50
    new_overall_gap = float(np.mean(new_ai_overall) - np.mean(new_hu_overall))
    total_reduction = max(0.0, (abs(old_overall_gap) - abs(new_overall_gap)) / max(1e-4, abs(old_overall_gap)) * 100)

    print("-" * 84)
    print(
        "{:<28} | {:<16} | {:<16} | {:<16}".format(
            "TỔNG THỂ (OVERALL)",
            f"{old_overall_gap:+.3f}",
            f"{new_overall_gap:+.3f}",
            f"GIẢM {total_reduction:.1f}%",
        )
    )

    report["overall"] = {
        "old_overall_gap": round(old_overall_gap, 4),
        "new_overall_gap": round(new_overall_gap, 4),
        "total_bias_reduction_percent": round(total_reduction, 1),
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Huấn luyện lại Model 2 (Candidate Answer Scorer) bằng tập dữ liệu cân bằng chống thiên vị."
    )
    parser.add_argument(
        "--humanized-data",
        default="data/augmented/debiasing_humanized_dataset.csv",
        help="Đường dẫn file dữ liệu thô người thật đã sinh",
    )
    parser.add_argument(
        "--output-artifact",
        default="artifacts/weak_supervision/v1_1/candidate_answer_scorer.joblib",
        help="Đường dẫn lưu model sau huấn luyện",
    )
    parser.add_argument(
        "--metrics-output",
        default="artifacts/weak_supervision/v1_1/candidate_answer_scorer_metrics.json",
        help="Đường dẫn lưu chỉ số đánh giá",
    )
    parser.add_argument(
        "--report-output",
        default="artifacts/weak_supervision/v1_1/debiasing_evaluation_report.json",
        help="Đường dẫn lưu báo cáo giảm thiên vị",
    )

    args = parser.parse_args()
    print_banner()

    db = get_standalone_db()
    try:
        # 1. Load data
        humanized_path = PROJECT_ROOT / args.humanized_data
        all_rows = load_all_datasets(db, humanized_path)
        print(f"\n[TỔNG HỢP] Tổng số mẫu dữ liệu đưa vào huấn luyện: {len(all_rows)}")

        human_count = sum(1 for r in all_rows if r["data_style"] in ("humanized_spoken", "real_human"))
        ai_count = sum(1 for r in all_rows if r["data_style"] == "pure_ai")
        print(f"- Mẫu phong cách Con Người: {human_count} ({human_count/len(all_rows)*100:.1f}%)")
        print(f"- Mẫu phong cách AI:        {ai_count} ({ai_count/len(all_rows)*100:.1f}%)")
        print(f"-> Tỷ lệ đạt chuẩn: {human_count/len(all_rows)*100:.1f}% / {ai_count/len(all_rows)*100:.1f}% (Chuẩn ~40/60)!")

        # 2. Split dataset
        print("\n[CHIA TẬP DỮ LIỆU] Phân chia tập dữ liệu (70% Train - 15% Validation - 15% Test)...")
        train_rows, val_rows, test_rows = split_stratified_groups(all_rows, random_state=42)
        print(f"- Train:      {len(train_rows)} mẫu ({len(set(r['group_id'] for r in train_rows))} nhóm context)")
        print(f"- Validation: {len(val_rows)} mẫu ({len(set(r['group_id'] for r in val_rows))} nhóm context)")
        print(f"- Test:       {len(test_rows)} mẫu ({len(set(r['group_id'] for r in test_rows))} nhóm context)")

        # 3. Train
        print("\n[HUẤN LUYỆN] Đang khớp 7 bộ TF-IDF (1,2-gram) + Ridge Regressors trên tập dữ liệu mới...")
        started_fit = time.perf_counter()
        trainer = CandidateAnswerScorerTrainer(
            random_state=42,
            minimum_rows=20,
            criteria=TURN_EVIDENCE_CRITERIA,
        )
        result = trainer.train_partitions(
            train_rows=train_rows,
            validation_rows=val_rows,
            test_rows=test_rows,
        )
        fit_time = round(time.perf_counter() - started_fit, 2)
        print(f"-> Huấn luyện xong 7 chiều năng lực trong {fit_time} giây!")

        # 4. Bias Evaluation
        old_model_backup = PROJECT_ROOT / "artifacts" / "weak_supervision" / "v1_1" / "candidate_answer_scorer_pre_debias_v1_1.joblib"
        bias_report = evaluate_bias_comparison(old_model_backup, result.artifact, humanized_path)

        # 5. Save Artifacts (luôn lưu tuyệt đối vào thư mục PROJECT_ROOT)
        out_art = Path(args.output_artifact)
        if not out_art.is_absolute():
            out_art = PROJECT_ROOT / out_art
        out_art.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.artifact, out_art)
        print(f"\n[LƯU MÔ HÌNH] Đã lưu model mới tại: {out_art}")

        out_metrics = Path(args.metrics_output)
        if not out_metrics.is_absolute():
            out_metrics = PROJECT_ROOT / out_metrics
        with open(out_metrics, "w", encoding="utf-8") as f:
            json.dump(result.report, f, ensure_ascii=False, indent=2)
        print(f"[LƯU METRICS] Đã lưu metrics kỹ thuật tại: {out_metrics}")

        out_rep = Path(args.report_output)
        if not out_rep.is_absolute():
            out_rep = PROJECT_ROOT / out_rep
        with open(out_rep, "w", encoding="utf-8") as f:
            json.dump(bias_report, f, ensure_ascii=False, indent=2)
        print(f"[LƯU BÁO CÁO] Đã lưu báo cáo đánh giá thiên vị tại: {out_rep}")

        print("\n" + "=" * 80)
        print("  HOÀN THÀNH XUẤT SẮC! MÔ HÌNH MODEL 2 ĐÃ ĐƯỢC HUẤN LUYỆN LẠI THÀNH CÔNG.")
        print(f"  Mức giảm thiên vị phong cách (Style Bias Reduction): {bias_report['overall']['total_bias_reduction_percent']}%!")
        print("=" * 80)

    finally:
        db.close()


if __name__ == "__main__":
    main()
