"""
FastAPI Application

Main application with lifespan management and middleware.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.backend.config import default_config
from app.routes import prediction, health, model, data, explain, monitoring, history, batch
from app.services.model_service import (
    init_model_service,
    ModelService,
    SyntheticModelRejectedError,
    ModelApprovalError,
    ModelNotLoadedError,
)
from app.services.feature_service import init_feature_service, FeatureService
from app.services.prediction_service import init_prediction_service, PredictionService
from src.monitoring.prediction_logger import PredictionLogger

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Startup:
    - Initialize services
    - Load production model (with safety validation)
    
    Shutdown:
    - Cleanup resources
    """
    # Startup
    logger.info("Starting AQI Predictor API...")
    
    try:
        # Initialize feature service
        feature_service = init_feature_service(
            fallback_enabled=default_config.feature_store_local_fallback_enabled,
        )
        
        # Initialize model service
        model_service = init_model_service(registry=None)
        
        # Initialize prediction logger
        prediction_logger = PredictionLogger(
            log_dir="data/predictions",
            enable_security_checks=True,
        )
        
        # Initialize prediction service
        prediction_service = init_prediction_service(
            model_service=model_service,
            feature_service=feature_service,
            prediction_logger=prediction_logger,
        )
        
        # Store services in app state
        app.state.model_service = model_service
        app.state.feature_service = feature_service
        app.state.prediction_service = prediction_service
        
        logger.info("Services initialized successfully")
        
        # Load production model from local file
        try:
            model_service.load_local_model()
            logger.info("Production model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load local model: {e}")
            logger.info("API will start without model - health check only")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Don't raise - allow app to start for health checks
    
    yield
    
    # Shutdown
    logger.info("Shutting down AQI Predictor API...")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title=default_config.app_name,
        version=default_config.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=default_config.cors_origins or ["*"],
        allow_methods=default_config.cors_methods or ["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Exception handlers
    @app.exception_handler(SyntheticModelRejectedError)
    async def handle_synthetic_rejected(request: Request, exc: SyntheticModelRejectedError):
        return JSONResponse(
            status_code=403,
            content={"detail": str(exc), "type": "SyntheticModelRejectedError"},
        )
    
    @app.exception_handler(ModelApprovalError)
    async def handle_approval_error(request: Request, exc: ModelApprovalError):
        return JSONResponse(
            status_code=403,
            content={"detail": str(exc), "type": "ModelApprovalError"},
        )
    
    @app.exception_handler(ModelNotLoadedError)
    async def handle_model_not_loaded(request: Request, exc: ModelNotLoadedError):
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "type": "ModelNotLoadedError"},
        )
    
    # Include routes
    app.include_router(prediction.router)
    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(data.router)
    app.include_router(explain.router)
    app.include_router(monitoring.router)
    app.include_router(history.router)
    app.include_router(batch.router)
    
    return app


# Create application instance
app = create_app()
