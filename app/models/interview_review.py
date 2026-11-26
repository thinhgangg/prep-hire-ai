from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class InterviewReview(Base):
    __tablename__ = "interview_reviews"

    review_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(
        String(36),
        ForeignKey("interview_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    shortlisted = Column(Boolean, nullable=False, default=False, index=True)
    decision = Column(String(30), nullable=False, default="pending", index=True)
    hr_notes = Column(Text, nullable=True)
    reviewed_by_user_id = Column(Integer, nullable=True)
    reviewed_by_email = Column(String(255), nullable=True)
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

    session = relationship("InterviewSession", back_populates="review")

