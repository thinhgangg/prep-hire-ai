from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class InterviewGazeReport(Base):
    __tablename__ = "interview_gaze_reports"

    report_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(
        String(36),
        ForeignKey("interview_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    total_frames = Column(Integer, nullable=False, default=0)
    face_frames = Column(Integer, nullable=False, default=0)
    duration_sec = Column(Float, nullable=False, default=0.0)
    camera_active = Column(Boolean, nullable=False, default=False)
    face_presence_pct = Column(Float, nullable=True)
    gaze_center_pct = Column(Float, nullable=True)
    gaze_left_pct = Column(Float, nullable=True)
    gaze_right_pct = Column(Float, nullable=True)
    gaze_away_pct = Column(Float, nullable=True)
    gaze_events_json = Column(JSON, nullable=False, default=list)
    gaze_timeline_json = Column(JSON, nullable=False, default=list)
    integrity_score = Column(Float, nullable=True)
    integrity_rank = Column(String(16), nullable=True)
    integrity_summary_json = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    session = relationship("InterviewSession", back_populates="gaze_report")

