from fastapi import APIRouter, HTTPException, status
from app.predictor import predictor
from app.schemas import ClassesResponse

router = APIRouter(tags=["Metadata"])


@router.get(
    "/classes",
    response_model=ClassesResponse,
    summary="List all class labels for each prediction task",
)
def classes() -> ClassesResponse:
    """
    Returns the ordered class labels for species (23), host (3), and geo (7) tasks.
    Index i in each list corresponds to model logit index i.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading.",
        )
    return ClassesResponse(
        species=predictor.label_maps["species"],
        host=predictor.label_maps["host"],
        geo=predictor.label_maps["geo"],
    )
