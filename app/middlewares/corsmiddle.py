from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

def add_cors_middleware(app: FastAPI):
    """
    Add CORSMiddleware

    Args:
    - app (FastAPI): FastAPI application instance.

    Returns:
    - app (FastAPI): FastAPI application instance with middleware.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"], 
        allow_credentials=True,
        allow_methods=["POST"] 
    )
    return app