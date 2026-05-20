"""
FastAPI application entry point.

Startup: predictor.load() blocks until model weights are on device.
Shutdown: nothing to clean up — PyTorch releases GPU memory when the process exits.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import RedirectResponse, Response

from app.config import settings
from app.middleware import RateLimitMiddleware
from app.predictor import predictor
from app.routers import classes, health, predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== HantaBERT-API starting up ===")
    predictor.load()
    yield
    logger.info("=== HantaBERT-API shutting down ===")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "Multi-task inference API for the HantaBERT model. "
        "Classifies hantavirus nucleotide sequences by species (23 classes), "
        "host (Rodent / Human / Others), and geographic origin (7 regions)."
    ),
    lifespan=lifespan,
    docs_url=None,
)

app.add_middleware(RateLimitMiddleware, limit=settings.rate_limit_per_minute)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(classes.router)


_CUSTOM_HEAD = b"""
<meta name="robots" content="none" />
<meta name="googlebot" content="none" />
<script defer src="https://stat.faizath.com/script.js" data-website-id="650fad88-8ee1-4974-9529-973e1d911b62" data-domains="hantabert-api.faizath.com"></script>
"""


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    html = get_swagger_ui_html(openapi_url=app.openapi_url, title=app.title)
    body = html.body.replace(b"</head>", _CUSTOM_HEAD + b"</head>")
    return Response(content=body, media_type="text/html")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
