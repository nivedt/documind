"""
Shared slowapi Limiter instance.

Import `limiter` wherever @limiter.limit() decorators are needed.
The app must set `app.state.limiter = limiter` on startup and register
the RateLimitExceeded handler — both done in app/main.py.
"""
from slowapi import Limiter
from slowapi.util import get_ipaddr

limiter = Limiter(key_func=get_ipaddr)
