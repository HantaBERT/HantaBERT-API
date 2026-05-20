import os
from pydantic_settings import BaseSettings, SettingsConfigDict

_HERE = os.path.dirname(os.path.abspath(__file__))          # HantaBERT-API/app/
_API_ROOT = os.path.dirname(_HERE)                           # HantaBERT-API/
_MODEL_DIR = os.path.join(_API_ROOT, "model")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model_weights_path: str = os.path.join(_MODEL_DIR, "best_model.pt")
    label_maps_path: str    = os.path.join(_MODEL_DIR, "label_maps.json")
    dnabert_model_path: str = "zhihan1996/DNABERT-2-117M"
    max_length: int         = 512
    device: str             = "auto"   # "auto" | "cpu" | "cuda" | "cuda:0"
    api_title: str          = "HantaBERT Inference API"
    api_version: str        = "1.0.0"
    rate_limit_per_minute: int = 10    # requests per IP per minute; 0 = disabled


settings = Settings()
