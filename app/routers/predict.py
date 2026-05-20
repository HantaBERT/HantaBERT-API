from fastapi import APIRouter, HTTPException, status
from app.predictor import predictor
from app.schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["Inference"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a hantavirus sequence by species, host, and geographic origin",
)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Accepts a raw DNA or RNA nucleotide sequence.

    - U bases are automatically converted to T.
    - Sequences longer than 512 BPE tokens are truncated (indicated in the response).
    - Returns per-task: predicted class, softmax confidence, and top-N alternatives.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading. Retry after /health returns model_loaded=true.",
        )
    if not request.sequence.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="sequence must not be empty.",
        )
    return predictor.predict(sequence=request.sequence, top_n=request.top_n)
