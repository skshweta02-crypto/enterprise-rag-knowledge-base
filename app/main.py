from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import logger
from app.api.health import router as health_router
from app.api.documents import router as documents_router
from app.api.query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} in {settings.APP_ENV} mode")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        description="Enterprise RAG Knowledge Base & Orchestration Service",
        lifespan=lifespan,
    )

    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(query_router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
