"""add interview gaze reports and hr reviews

Revision ID: b9d4e7f2c1a3
Revises: a8c3d6e1f4b9
Create Date: 2026-11-26 16:05:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b9d4e7f2c1a3"
down_revision: Union[str, None] = "a8c3d6e1f4b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "interview_gaze_reports",
        sa.Column("report_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("total_frames", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("face_frames", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("duration_sec", sa.Float(), nullable=False, server_default="0"),
        sa.Column("camera_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("face_presence_pct", sa.Float(), nullable=True),
        sa.Column("gaze_center_pct", sa.Float(), nullable=True),
        sa.Column("gaze_left_pct", sa.Float(), nullable=True),
        sa.Column("gaze_right_pct", sa.Float(), nullable=True),
        sa.Column("gaze_away_pct", sa.Float(), nullable=True),
        sa.Column("gaze_events_json", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("gaze_timeline_json", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("integrity_score", sa.Float(), nullable=True),
        sa.Column("integrity_rank", sa.String(length=16), nullable=True),
        sa.Column("integrity_summary_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["interview_sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("report_id"),
        sa.UniqueConstraint("session_id", name="uq_interview_gaze_reports_session_id"),
    )
    op.create_index(
        "ix_interview_gaze_reports_session_id",
        "interview_gaze_reports",
        ["session_id"],
    )

    op.create_table(
        "interview_reviews",
        sa.Column("review_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("shortlisted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("decision", sa.String(length=30), nullable=False, server_default="pending"),
        sa.Column("hr_notes", sa.Text(), nullable=True),
        sa.Column("reviewed_by_user_id", sa.Integer(), nullable=True),
        sa.Column("reviewed_by_email", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["interview_sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("review_id"),
        sa.UniqueConstraint("session_id", name="uq_interview_reviews_session_id"),
    )
    op.create_index(
        "ix_interview_reviews_session_id",
        "interview_reviews",
        ["session_id"],
    )
    op.create_index(
        "ix_interview_reviews_shortlisted",
        "interview_reviews",
        ["shortlisted"],
    )


def downgrade() -> None:
    op.drop_index("ix_interview_reviews_shortlisted", table_name="interview_reviews")
    op.drop_index("ix_interview_reviews_session_id", table_name="interview_reviews")
    op.drop_table("interview_reviews")
    op.drop_index("ix_interview_gaze_reports_session_id", table_name="interview_gaze_reports")
    op.drop_table("interview_gaze_reports")

