from fastapi import Request, HTTPException
import jwt
import os

JWT_SECRET = os.getenv("JWT_SECRET")
ADMIN_KEY = os.getenv("ADMIN_KEY")


def require_admin(request: Request):
    # ✅ Check secret admin header
    admin_key = request.headers.get("X-Admin-Key")
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    # ✅ Check cookie token
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="No token provided")

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

    # ✅ Check role
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    return payload
