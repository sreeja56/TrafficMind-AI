"""
TrafficMind AI - Root Entry Point
Exports the FastAPI 'app' ASGI callable for Vercel Serverless runtime,
resolving Vercel's 'Found app.py but it does not export a top-level function' error.
"""

from api.index import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
