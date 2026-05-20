from fastapi import APIRouter
from app.predictor import predictor
from app.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Readiness check")
def health() -> HealthResponse:
    """Returns whether the model has finished loading. Poll before sending /predict requests."""
    return HealthResponse(
        status="ok" if predictor.is_loaded else ("error" if predictor.load_error else "loading"),
        model_loaded=predictor.is_loaded,
        device=str(predictor.device) if predictor.device is not None else "unknown",
    )
