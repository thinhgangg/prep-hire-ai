from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.base_class import Base
from app.db.session import get_db
from app.main import app
from app.models.interview_gaze_report import InterviewGazeReport
from app.models.interview_review import InterviewReview
from app.models.interview_session import InterviewSession
from app.models.interview_turn import InterviewTurn
from app.models.user_data import User


class InterviewAuthGazeReviewApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(
            self.engine,
            tables=[
                User.__table__,
                InterviewSession.__table__,
                InterviewTurn.__table__,
                InterviewGazeReport.__table__,
                InterviewReview.__table__,
            ],
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        def override_get_db():
            db = self.SessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self._old_internal = settings.INTERNAL_SERVICE_SECRET
        self._old_enforced = settings.AUTH_ENFORCED
        settings.INTERNAL_SERVICE_SECRET = "ai-test-internal-secret"
        settings.AUTH_ENFORCED = True

        db = self.SessionLocal()
        now = datetime.now(timezone.utc)
        db.add(
            User(
                user_id=10,
                email="duy.nguyen@prephire.vn",
                name="Nguyễn Đình Duy",
                role="user",
                created_at=now,
            )
        )
        db.add(
            InterviewSession(
                session_id="sess-owner-10",
                user_id=10,
                status="completed",
                language="vi",
                max_questions=5,
                current_turn_index=0,
                current_stage="completed",
                allow_external_ai=False,
                research_consent=False,
                context_source="user_upload",
                cv_text="Python FastAPI React developer with 3 years experience",
                jd_text="Seeking Backend Engineer proficient in Python and PostgreSQL",
                cv_profile_json={"name": "Nguyen Van A"},
                job_profile_json={"title": "Backend Engineer"},
                created_at=now,
                updated_at=now,
                completed_at=now,
            )
        )
        db.commit()
        db.close()

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        settings.INTERNAL_SERVICE_SECRET = self._old_internal
        settings.AUTH_ENFORCED = self._old_enforced
        self.engine.dispose()

    def _req(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def _call() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(_call())

    def _gateway_headers(self, user_id: int, role: str) -> dict[str, str]:
        return {
            "X-Internal-Service-Key": "ai-test-internal-secret",
            "X-User-Id": str(user_id),
            "X-User-Role": role,
            "X-User-Email": f"user{user_id}@prephire.vn",
        }

    def test_unauthenticated_blocked(self) -> None:
        self.assertEqual(self._req("GET", "/api/v1/interview/sessions").status_code, 401)
        self.assertEqual(
            self._req("GET", "/api/v1/interview/sessions/sess-owner-10").status_code,
            401,
        )

    def test_idor_blocks_other_candidate(self) -> None:
        other_headers = self._gateway_headers(20, "CANDIDATE")
        self.assertEqual(
            self._req(
                "GET", "/api/v1/interview/sessions/sess-owner-10", headers=other_headers
            ).status_code,
            403,
        )
        self.assertEqual(
            self._req(
                "GET", "/api/v1/interview/users/10/sessions", headers=other_headers
            ).status_code,
            403,
        )
        self.assertEqual(
            self._req("GET", "/api/v1/interview/sessions", headers=other_headers).status_code,
            403,
        )

    def test_gaze_report_persistence_and_cross_device_admin_read(self) -> None:
        owner_headers = self._gateway_headers(10, "CANDIDATE")
        admin_headers = self._gateway_headers(1, "ADMIN")

        put_res = self._req(
            "PUT",
            "/api/v1/interview/sessions/sess-owner-10/gaze",
            headers=owner_headers,
            json={
                "total_frames": 200,
                "face_frames": 190,
                "duration_sec": 65.4,
                "camera_active": True,
                "face_presence_pct": 95.0,
                "gaze_center_pct": 88.5,
                "gaze_left_pct": 4.0,
                "gaze_right_pct": 2.5,
                "gaze_away_pct": 6.5,
                "gaze_events": [
                    {
                        "timestamp": "00:22",
                        "direction": "NHIN TRAI",
                        "duration": 2.6,
                        "behavior": "SIDE_OR_DOWN",
                        "penalty": 3.9,
                        "question_index": 2,
                    }
                ],
                "gaze_timeline": [],
                "integrity_score": 85.8,
                "integrity_rank": "HIGH",
                "integrity_summary": {
                    "overallScore": 85.8,
                    "rank": "HIGH",
                    "hasSecondPerson": False,
                    "totalQuestionsScored": 1,
                },
            },
        )
        self.assertEqual(put_res.status_code, 200, put_res.text)

        get_res = self._req(
            "GET",
            "/api/v1/interview/sessions/sess-owner-10/gaze",
            headers=admin_headers,
        )
        self.assertEqual(get_res.status_code, 200, get_res.text)
        body = get_res.json()
        self.assertTrue(body["camera_active"])
        self.assertEqual(body["face_presence_pct"], 95.0)
        self.assertEqual(body["integrity_score"], 85.8)
        self.assertEqual(body["integrity_rank"], "HIGH")
        self.assertEqual(len(body["gaze_events"]), 1)

    def test_hr_review_and_shortlist_staff_only(self) -> None:
        owner_headers = self._gateway_headers(10, "CANDIDATE")
        admin_headers = self._gateway_headers(1, "ADMIN")

        self.assertEqual(
            self._req(
                "PUT",
                "/api/v1/interview/sessions/sess-owner-10/review",
                headers=owner_headers,
                json={"shortlisted": True},
            ).status_code,
            403,
        )

        res = self._req(
            "PUT",
            "/api/v1/interview/sessions/sess-owner-10/review",
            headers=admin_headers,
            json={
                "shortlisted": True,
                "decision": "interview_next",
                "hr_notes": "Rất vững kiến thức FastAPI & SQL",
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertTrue(res.json()["shortlisted"])
        self.assertEqual(res.json()["decision"], "interview_next")

        sess = self._req(
            "GET",
            "/api/v1/interview/sessions/sess-owner-10",
            headers=admin_headers,
        )
        self.assertEqual(sess.status_code, 200)
        self.assertEqual(sess.json()["candidate_name"], "Nguyễn Đình Duy")
        self.assertEqual(sess.json()["candidate_email"], "duy.nguyen@prephire.vn")
        self.assertTrue(sess.json()["review"]["shortlisted"])
        self.assertEqual(
            sess.json()["review"]["hr_notes"], "Rất vững kiến thức FastAPI & SQL"
        )

    def test_delete_session_owner_and_idor(self) -> None:
        other_headers = self._gateway_headers(20, "CANDIDATE")
        owner_headers = self._gateway_headers(10, "CANDIDATE")

        self.assertEqual(
            self._req(
                "DELETE",
                "/api/v1/interview/sessions/sess-owner-10",
                headers=other_headers,
            ).status_code,
            403,
        )

        deleted = self._req(
            "DELETE",
            "/api/v1/interview/sessions/sess-owner-10",
            headers=owner_headers,
        )
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertEqual(deleted.json()["status"], "deleted")

        after = self._req(
            "GET",
            "/api/v1/interview/sessions/sess-owner-10",
            headers=owner_headers,
        )
        self.assertEqual(after.status_code, 404)
