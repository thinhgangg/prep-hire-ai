from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Optional

import jwt
from fastapi import Depends, Header, HTTPException, status

from app.core.config import settings

STAFF_ROLES = {"ADMIN", "RECRUITER"}


@dataclass(frozen=True)
class RequestActor:
    user_id: Optional[int]
    role: str
    email: str = ""
    is_internal_gateway: bool = False

    @property
    def is_staff(self) -> bool:
        return self.role.upper() in STAFF_ROLES


def _normalize_role(raw: Optional[str]) -> str:
    value = (raw or "").strip().upper()
    if value in {"ADMIN", "RECRUITER", "CANDIDATE"}:
        return value
    if value == "USER":
        return "CANDIDATE"
    return "CANDIDATE"


def _verify_internal_key(provided: Optional[str]) -> bool:
    expected = (settings.INTERNAL_SERVICE_SECRET or "").strip()
    if not expected or not provided:
        return False
    return secrets.compare_digest(provided.strip(), expected)


def _decode_jwt(token: str) -> dict:
    secrets_to_try = [
        s.strip()
        for s in (settings.JWT_SECRET, settings.JWT_ACCESS_SECRET)
        if s and s.strip()
    ]
    last_err: Exception | None = None
    for secret in secrets_to_try:
        try:
            return jwt.decode(token, secret, algorithms=[settings.JWT_ALGORITHM])
        except jwt.ExpiredSignatureError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Phiên đăng nhập đã hết hạn.",
            ) from exc
        except jwt.InvalidTokenError as exc:
            last_err = exc
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token xác thực không hợp lệ.",
    ) from last_err


def get_request_actor(
    authorization: Optional[str] = Header(None),
    x_internal_service_key: Optional[str] = Header(None, alias="X-Internal-Service-Key"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    x_user_role: Optional[str] = Header(None, alias="X-User-Role"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
) -> RequestActor:
    """Defense-in-depth auth: accepts verified Gateway headers OR direct signed JWT."""
    if _verify_internal_key(x_internal_service_key):
        uid = int(x_user_id) if (x_user_id and x_user_id.isdigit()) else None
        return RequestActor(
            user_id=uid,
            role=_normalize_role(x_user_role),
            email=(x_user_email or "").strip(),
            is_internal_gateway=True,
        )

    if authorization:
        parts = authorization.strip().split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
            payload = _decode_jwt(parts[1].strip())
            sub = str(payload.get("sub") or payload.get("user_id_num") or "").strip()
            uid = int(sub) if sub.isdigit() else None
            return RequestActor(
                user_id=uid,
                role=_normalize_role(payload.get("role")),
                email=str(payload.get("email") or "").strip(),
            )

    if not settings.AUTH_ENFORCED:
        return RequestActor(user_id=None, role="ADMIN", is_internal_gateway=False)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Yêu cầu xác thực để truy cập AI service.",
    )


def require_staff(
    actor: RequestActor = Depends(get_request_actor),
) -> RequestActor:
    if not actor.is_staff:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Chỉ Quản trị viên hoặc Nhà tuyển dụng mới có quyền thực hiện thao tác này.",
        )
    return actor


def assert_session_access(session, actor: RequestActor) -> None:
    """Prevent IDOR: Candidates can only access sessions belonging to them."""
    if actor.is_staff:
        return
    session_uid = getattr(session, "user_id", None)
    if session_uid is not None and actor.user_id is not None and session_uid != actor.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bạn không có quyền truy cập phiên phỏng vấn của người khác.",
        )

