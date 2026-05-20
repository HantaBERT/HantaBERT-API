from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sequence: str = Field(
        ...,
        description="DNA or RNA nucleotide sequence. U bases are auto-converted to T.",
        examples=["ATGAAAGACCTTCTGAAGAAATTTGAG"],
    )
    top_n: int = Field(
        default=3,
        ge=1,
        le=23,
        description="Number of top predictions to return per task (including the top prediction).",
    )


class ClassScore(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class TaskPrediction(BaseModel):
    predicted: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_n: List[ClassScore]


class PredictResponse(BaseModel):
    species: TaskPrediction
    host: TaskPrediction
    geo: TaskPrediction
    sequence_length: int
    truncated: bool = Field(
        description="True if the input sequence exceeded 512 BPE tokens and was truncated."
    )


class HealthResponse(BaseModel):
    status: str        # "ok" | "loading" | "error"
    model_loaded: bool
    device: str


class ClassesResponse(BaseModel):
    species: List[str]
    host: List[str]
    geo: List[str]
