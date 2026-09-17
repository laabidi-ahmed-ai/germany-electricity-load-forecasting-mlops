"""FastAPI serving app (README §15, Phase 4).

Endpoints
---------
* ``GET /health``          - liveness + which champion is loaded + DB freshness
* ``GET /forecast``        - compute the day-ahead forecast *now* with the champion
                             model and the training feature pipeline (``?hours=24``)
* ``GET /forecast/latest`` - the most recently issued batch forecast from the DB
                             (instant; what the dashboard reads)

The champion is loaded once at startup from the database export
(``model_artifacts``, see ``src.models.registry``) - the API needs only ``DATABASE_URL``. ``create_app(state_loader=...)`` lets tests
inject a fake model / SQLite engine without touching MLflow or the network.

Run locally: ``uvicorn src.serving.api:app --reload`` (``make serve``).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.engine import Engine

from config.settings import get_settings
from src.data import db
from src.data.weather_client import WeatherClient
from src.models.registry import LoadedModel, load_champion
from src.serving.forecast import DAY_AHEAD_HOURS, NoDataError, make_day_ahead_forecast

log = logging.getLogger(__name__)

API_VERSION = "0.1.0"


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
@dataclass
class AppState:
    engine: Engine
    model: LoadedModel
    weather_client: WeatherClient | None
    started_at: datetime


def default_state_loader() -> AppState:
    engine = db.get_engine()
    db.init_db(engine)
    model = load_champion(engine)
    return AppState(
        engine=engine,
        model=model,
        weather_client=WeatherClient(),
        started_at=datetime.now(UTC),
    )


def get_state(request: Request) -> AppState:
    state = getattr(request.app.state, "ctx", None)
    if state is None:
        raise HTTPException(status_code=503, detail="service is starting up")
    return state


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ModelInfo(BaseModel):
    name: str
    version: str
    alias: str | None
    horizon: str
    n_features: int
    features: list[str]


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    api_version: str
    started_at: datetime
    model: ModelInfo
    database: str
    last_actual_utc: datetime | None
    last_forecast_issued_utc: datetime | None


class ForecastPoint(BaseModel):
    timestamp_utc: datetime
    forecast_mw: float


class ForecastResponse(BaseModel):
    model: ModelInfo
    horizon: str
    issued_at: datetime
    last_actual_utc: datetime
    target_start_utc: datetime
    target_end_utc: datetime
    forecast: list[ForecastPoint]


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(state_loader: Callable[[], AppState] = default_state_loader) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ctx = state_loader()
        log.info("champion loaded: %s", app.state.ctx.model.version_label)
        yield

    app = FastAPI(
        title="Germany electricity load forecast API",
        version=API_VERSION,
        description="Day-ahead hourly national load forecast (DE_LU) from the champion model.",
        lifespan=lifespan,
    )

    def model_info(model: LoadedModel) -> ModelInfo:
        return ModelInfo(
            name=model.name,
            version=model.version,
            alias=model.alias,
            horizon=model.horizon.name,
            n_features=len(model.features),
            features=model.features,
        )

    @app.get("/health", response_model=HealthResponse)
    def health(state: AppState = Depends(get_state)) -> HealthResponse:
        last_actual = db.latest_timestamp(state.engine, db.LoadActual)
        latest_fc = db.read_latest_model_forecast(state.engine)
        issued = None if latest_fc.empty else latest_fc["issued_at"].iloc[0].to_pydatetime()
        return HealthResponse(
            status="ok",
            api_version=API_VERSION,
            started_at=state.started_at,
            model=model_info(state.model),
            database=state.engine.url.get_backend_name(),
            last_actual_utc=None if last_actual is None else last_actual.to_pydatetime(),
            last_forecast_issued_utc=issued,
        )

    @app.get("/forecast", response_model=ForecastResponse)
    def forecast(
        hours: int = Query(DAY_AHEAD_HOURS, ge=1, le=DAY_AHEAD_HOURS),
        state: AppState = Depends(get_state),
    ) -> ForecastResponse:
        try:
            result = make_day_ahead_forecast(
                state.engine, state.model, weather_client=state.weather_client, hours=hours
            )
        except NoDataError as err:
            raise HTTPException(status_code=503, detail=str(err)) from err
        return ForecastResponse(
            model=model_info(state.model),
            horizon=result.horizon,
            issued_at=result.issued_at.to_pydatetime(),
            last_actual_utc=result.last_actual.to_pydatetime(),
            target_start_utc=result.target_start.to_pydatetime(),
            target_end_utc=result.target_end.to_pydatetime(),
            forecast=[
                ForecastPoint(timestamp_utc=ts.to_pydatetime(), forecast_mw=round(float(v), 1))
                for ts, v in result.frame["forecast_mw"].items()
            ],
        )

    @app.get("/forecast/latest", response_model=ForecastResponse)
    def forecast_latest(state: AppState = Depends(get_state)) -> ForecastResponse:
        latest = db.read_latest_model_forecast(state.engine)
        if latest.empty:
            raise HTTPException(status_code=404, detail="no batch forecast stored yet")
        last_actual = db.latest_timestamp(state.engine, db.LoadActual) or pd.Timestamp(0, tz="UTC")
        version = str(latest["model_version"].iloc[0])
        info = model_info(state.model)
        if version != state.model.version:
            info = info.model_copy(update={"version": version, "alias": None})
        return ForecastResponse(
            model=info,
            horizon=state.model.horizon.name,
            issued_at=latest["issued_at"].iloc[0].to_pydatetime(),
            last_actual_utc=last_actual.to_pydatetime(),
            target_start_utc=latest["timestamp_utc"].min().to_pydatetime(),
            target_end_utc=latest["timestamp_utc"].max().to_pydatetime(),
            forecast=[
                ForecastPoint(
                    timestamp_utc=r.timestamp_utc.to_pydatetime(),
                    forecast_mw=round(r.forecast_mw, 1),
                )
                for r in latest.sort_values("timestamp_utc").itertuples(index=False)
            ],
        )

    return app


logging.basicConfig(level=get_settings().log_level)
app = create_app()
