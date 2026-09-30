from fastapi import APIRouter
from pydantic import BaseModel
from app.core.config import settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV,
    )
