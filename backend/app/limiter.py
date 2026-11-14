from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

def get_user_rate_limit_key(request: Request) -> str:
    # Use Authorization header for per-user limiting, fallback to client IP
    auth_header = request.headers.get("Authorization")
    if auth_header:
        return auth_header
    return get_remote_address(request)

limiter = Limiter(key_func=get_user_rate_limit_key, default_limits=[])
