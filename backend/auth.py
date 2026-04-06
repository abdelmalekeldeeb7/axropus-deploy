from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sqlalchemy.orm import Session

try:
    from .config import get_settings
    from .db import get_db
    from .models import APIKey, Customer
    from .rate_limit import enforce_rate_limit
except ImportError:
    from config import get_settings
    from db import get_db
    from models import APIKey, Customer
    from rate_limit import enforce_rate_limit

router = APIRouter(prefix="/api/auth", tags=["auth"])
bearer = HTTPBearer(auto_error=False)


class SignupRequest(BaseModel):
    email: str
    password: str
    company_name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(customer_id: int) -> str:
    settings = get_settings()
    expire = _utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": str(customer_id),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


def get_current_customer(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: Session = Depends(get_db),
) -> Customer:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    claims = decode_access_token(credentials.credentials)
    sub = claims.get("sub")
    try:
        customer_id = int(str(sub))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject") from exc
    customer = db.get(Customer, customer_id)
    if customer is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Customer not found")
    return customer


@router.post("/signup")
def signup(payload: SignupRequest, request: Request, db: Session = Depends(get_db)) -> dict:
    settings = get_settings()
    email = payload.email.strip().lower()
    client_host = request.client.host if request.client else "unknown"
    enforce_rate_limit(
        key=f"signup:{client_host}:{email}",
        limit=settings.signup_rate_limit_per_hour,
        window_seconds=3600,
    )

    if not email or "@" not in email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email")
    if len(payload.password or "") < 8:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must be at least 8 characters")

    existing = db.query(Customer).filter(Customer.email == email).first()
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    customer = Customer(
        email=email,
        password_hash=hash_password(payload.password),
        company_name=(payload.company_name.strip() if payload.company_name else None),
    )
    db.add(customer)
    db.flush()

    trial_key_value = f"ax-{secrets.token_hex(16)}"
    trial_expiry = _utcnow().replace(tzinfo=None) + timedelta(days=settings.trial_key_days)
    api_key = APIKey(
        customer_id=customer.id,
        key=trial_key_value,
        status="trial",
        tier="trial",
        expires_at=trial_expiry,
    )
    db.add(api_key)
    db.commit()
    db.refresh(customer)

    token = create_access_token(customer.id)
    return {
        "customer_id": customer.id,
        "api_key": trial_key_value,
        "token": token,
    }


@router.post("/login")
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)) -> dict:
    email = payload.email.strip().lower()
    settings = get_settings()
    client_host = request.client.host if request.client else "unknown"
    enforce_rate_limit(
        key=f"login:{client_host}:{email}",
        limit=settings.login_rate_limit_per_minute,
        window_seconds=60,
    )

    customer = db.query(Customer).filter(Customer.email == email).first()
    if customer is None or not verify_password(payload.password, customer.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(customer.id)
    return {
        "customer_id": customer.id,
        "token": token,
    }


@router.get("/me")
def me(customer: Customer = Depends(get_current_customer)) -> dict:
    return {
        "customer_id": customer.id,
        "email": customer.email,
        "company_name": customer.company_name,
        "created_at": customer.created_at,
    }
