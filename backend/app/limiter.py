from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from backend.app.auth_utils import decode_access_token

def get_user_rate_limit_key(request: Request) -> str:
    # Decode user_id from JWT in Authorization header for per-user rate limiting;
    # fallback to client remote IP if token is missing or invalid.
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        payload = decode_access_token(token)
        user_id = payload.get("user_id")
        if user_id is not None:
            return f"user:{user_id}"
    return get_remote_address(request)

limiter = Limiter(key_func=get_user_rate_limit_key, default_limits=[])
