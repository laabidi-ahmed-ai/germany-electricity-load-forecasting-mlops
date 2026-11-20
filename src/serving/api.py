"""FastAPI serving app.

* ``GET /health``          - liveness, which champion is loaded, data freshness, and
                             ``status`` = ``ok`` | ``degraded`` with short ``reasons``
* ``GET /forecast``        - compute the day-ahead forecast now with the champion
                             and the training feature pipeline (``?hours=24``)
* ``GET /forecast/latest`` - the most recently issued batch forecast from the DB
                             (instant; what the dashboard reads), described by the model
                             that issued it

The champion is loaded at startup from the database export (``model_artifacts``, see
``src.models.registry``), so the API needs only ``DATABASE_URL``. It then follows the
champion: at most once per ``ServingConfig.status_ttl_seconds`` a request reads the champion
pointer and the data ages in one query (``db.serving_status``); a changed pointer (version
or bundle SHA) reloads the model in a background thread while every request keeps serving
the old one. The new model, its version and SHA are swapped in as one immutable
``ServedModel``, so no request mixes two models. A failed reload keeps the old model and
shows in ``/health``.

``/health`` always answers HTTP 200: stale data is not a dead process, and the image's
``curl -f`` healthcheck would otherwise mark a working container unhealthy (and get it
restarted by anything that restarts unhealthy containers). ``reasons`` are fixed labels,
never exception text (``/health`` is public; DB errors can carry connection details) -
the full error goes to the log.

``create_app(state_loader=..., config=...)`` lets tests inject a fake model, a SQLite
engine, a clock and synchronous reloads.

Run locally: ``uvicorn src.serving.api:app --reload`` (``make serve``).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import InitVar, dataclass, field
from datetime import UTC, datetime

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.engine import Engine

from config.log import configure_logging
from src.data import db
from src.data.db import SOURCE_SMARD
from src.data.weather_client import WeatherClient
from src.models.registry import CorruptBundleError, LoadedModel, load_champion
from src.serving.forecast import DAY_AHEAD_HOURS, NoDataError, make_day_ahead_forecast

log = logging.getLogger(__name__)

API_VERSION = "0.2.0"

STATUS_TTL_SECONDS = 60.0  # champion pointer + data ages are re-read at most this often
# Ingest runs hourly and SMARD publishes ~1-2 h late: a normal age is under 3 h.
MAX_ACTUAL_AGE_HOURS = 6.0
# Same threshold as the Datadog monitor "No forecast today" (docs/observability.md), so
# "stale forecast" means the same everywhere: one issue a day at ~05:40 UTC, plus slack.
MAX_FORECAST_AGE_HOURS = 25.0
# A failed reload of the same pointer is retried this often (each try downloads the bundle);
# a CorruptBundleError is never retried for the same SHA - it will not load, ever.
RELOAD_RETRY_SECONDS = 600.0
# Opening a DB connection may take this long (libpq ``connect_timeout``), so a slow wake-up
# gives "database unreachable" well inside the healthcheck's 5 s timeout.
DB_CONNECT_TIMEOUT_SECONDS = 3

# /health reasons: fixed labels only (see module docstring).
REASON_DB_UNREACHABLE = "database unreachable"
REASON_NO_ACTUALS = "no actual load stored"
REASON_NO_FORECAST = "no forecast issued yet"
REASON_NO_CHAMPION = "no champion in database"
REASON_RELOAD_IN_PROGRESS = "champion reload in progress"
REASON_OLD_CHAMPION = "serving an older champion"
REASON_STATUS_PENDING = "status check in progress"


def _utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


@dataclass(frozen=True)
class ServingConfig:
    status_ttl_seconds: float = STATUS_TTL_SECONDS
    max_actual_age_hours: float = MAX_ACTUAL_AGE_HOURS
    max_forecast_age_hours: float = MAX_FORECAST_AGE_HOURS
    reload_retry_seconds: float = RELOAD_RETRY_SECONDS
    background_reload: bool = True  # tests run reloads inline
    now: Callable[[], pd.Timestamp] = _utcnow  # wall clock for data ages
    monotonic: Callable[[], float] = time.monotonic  # clock for the TTL and retry waits


# --- State ---
@dataclass(frozen=True)
class ServedModel:
    """The model being served with the pointer it was loaded for. Replaced, never mutated."""

    model: LoadedModel
    loaded_at: datetime

    @property
    def pointer(self) -> tuple[str, str | None]:
        return self.model.version, self.model.bundle_sha256

    @classmethod
    def of(cls, model: LoadedModel) -> ServedModel:
        return cls(model=model, loaded_at=datetime.now(UTC))


@dataclass(frozen=True)
class StatusSnapshot:
    """``db.serving_status`` at one moment, or the fact that the database did not answer."""

    checked_at: pd.Timestamp
    checked_monotonic: float
    status: db.ServingStatus | None  # None: database unreachable

    @property
    def pointer(self) -> tuple[str | None, str | None]:
        if self.status is None:
            return None, None
        return self.status.champion_version, self.status.champion_sha256


@dataclass(frozen=True)
class ReloadFailure:
    pointer: tuple[str | None, str | None]
    error: str  # exception class name only
    at_monotonic: float
    permanent: bool  # CorruptBundleError: never retry this pointer

    @property
    def reason(self) -> str:
        return f"champion reload failed ({self.error})"


@dataclass
class AppState:
    """Engine, the served champion and the watcher that follows the DB champion pointer.

    Read ``served`` once per request into a local: it may be replaced between two reads.
    """

    engine: Engine
    model: InitVar[LoadedModel]
    weather_client: WeatherClient | None
    started_at: datetime
    model_loader: Callable[[Engine], LoadedModel] = load_champion
    config: ServingConfig = field(default_factory=ServingConfig)
    served: ServedModel = field(init=False)
    snapshot: StatusSnapshot | None = field(init=False, default=None)
    reload_failure: ReloadFailure | None = field(init=False, default=None)
    reload_thread: threading.Thread | None = field(init=False, default=None)
    _status_lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _reload_lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self, model: LoadedModel) -> None:
        self.served = ServedModel.of(model)

    @property
    def champion_name(self) -> str:
        return self.served.model.name

    @property
    def reloading(self) -> bool:
        return self._reload_lock.locked()

    # -- status snapshot (TTL-cached, one DB round trip) --
    def current_snapshot(self, *, force: bool = False) -> StatusSnapshot | None:
        """The cached snapshot, re-read when older than the TTL. Never blocks on another
        request's refresh: while one is running, the previous snapshot is returned."""
        snap = self.snapshot
        fresh = (
            snap is not None
            and self.config.monotonic() - snap.checked_monotonic < self.config.status_ttl_seconds
        )
        if fresh and not force:
            return snap
        if not self._status_lock.acquire(blocking=False):
            return snap
        try:
            try:
                status: db.ServingStatus | None = db.serving_status(self.engine, self.champion_name)
            except Exception:
                log.exception("health query failed - reporting the database as unreachable")
                status = None
            snap = StatusSnapshot(
                checked_at=self.config.now(),
                checked_monotonic=self.config.monotonic(),
                status=status,
            )
            self.snapshot = snap
        finally:
            self._status_lock.release()
        return snap

    # -- champion reload --
    def follow_champion(self) -> StatusSnapshot | None:
        """Refresh the snapshot if due and start a reload when the DB pointer moved."""
        snap = self.current_snapshot()
        if snap is None or snap.status is None or snap.status.champion_version is None:
            return snap
        if snap.pointer == self.served.pointer:
            return snap
        failure = self.reload_failure
        if failure is not None and failure.pointer == snap.pointer:
            waited = self.config.monotonic() - failure.at_monotonic
            if failure.permanent or waited < self.config.reload_retry_seconds:
                return snap
        if not self._reload_lock.acquire(blocking=False):
            return snap  # another request is reloading; keep serving the old model
        if self.config.background_reload:
            self.reload_thread = threading.Thread(
                target=self._reload, args=(snap.pointer,), name="champion-reload", daemon=True
            )
            self.reload_thread.start()
        else:
            self._reload(snap.pointer)
        return snap

    def _reload(self, pointer: tuple[str | None, str | None]) -> None:
        """Load the DB champion and swap it in. Runs with ``_reload_lock`` held."""
        try:
            old = self.served.model.version_label
            model = self.model_loader(self.engine)
            self.served = ServedModel.of(model)  # one assignment: model, version and SHA
            self.reload_failure = None
            log.info("champion reloaded: %s -> %s", old, model.version_label)
        except Exception as err:
            log.exception(
                "champion reload failed for version %s - keeping the old model", pointer[0]
            )
            self.reload_failure = ReloadFailure(
                pointer=pointer,
                error=type(err).__name__,
                at_monotonic=self.config.monotonic(),
                permanent=isinstance(err, CorruptBundleError),
            )
        finally:
            self._reload_lock.release()


def default_state_loader() -> AppState:
    configure_logging()
    engine = db.get_engine(connect_timeout=DB_CONNECT_TIMEOUT_SECONDS)
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
    state.follow_champion()
    return state


# --- Schemas ---
class ModelInfo(BaseModel):
    name: str
    version: str
    alias: str | None
    horizon: str
    n_features: int | None = Field(
        description="None when the version's export was pruned and its features are unknown"
    )
    features: list[str] | None


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"], description="ok | degraded")
    reasons: list[str] = Field(examples=[["latest forecast older than 25 h"]])
    api_version: str
    started_at: datetime
    model: ModelInfo
    champion_db_version: str | None
    database: str = Field(description="backend name, or 'unreachable'")
    checked_at: datetime | None = Field(description="when the DB status below was read")
    last_actual_utc: datetime | None
    last_actual_age_hours: float | None
    last_forecast_issued_utc: datetime | None
    last_forecast_age_hours: float | None


class ForecastPoint(BaseModel):
    timestamp_utc: datetime
    forecast_mw: float


def forecast_points(forecast_mw: pd.Series) -> list[ForecastPoint]:
    """``forecast_mw`` indexed by UTC timestamp -> response points (MW rounded to 0.1)."""
    return [
        ForecastPoint(timestamp_utc=ts.to_pydatetime(), forecast_mw=round(float(v), 1))
        for ts, v in forecast_mw.items()
    ]


class ForecastResponse(BaseModel):
    model: ModelInfo
    horizon: str
    issued_at: datetime
    last_actual_utc: datetime
    target_start_utc: datetime
    target_end_utc: datetime
    forecast: list[ForecastPoint]


def model_info(model: LoadedModel) -> ModelInfo:
    return ModelInfo(
        name=model.name,
        version=model.version,
        alias=model.alias,
        horizon=model.horizon.name,
        n_features=len(model.features),
        features=model.features,
    )


def _age_hours(ts: pd.Timestamp | None, now: pd.Timestamp) -> float | None:
    return None if ts is None else round((now - ts).total_seconds() / 3600, 2)


def health_reasons(state: AppState, snap: StatusSnapshot | None) -> list[str]:
    """Why the service is degraded (empty = ok). Fixed labels only."""
    cfg = state.config
    reasons: list[str] = []
    if snap is None:
        return [REASON_STATUS_PENDING]
    status = snap.status
    if status is None:
        reasons.append(REASON_DB_UNREACHABLE)
    else:
        now = cfg.now()
        actual_age = _age_hours(status.last_actual, now)
        if actual_age is None:
            reasons.append(REASON_NO_ACTUALS)
        elif actual_age > cfg.max_actual_age_hours:
            reasons.append(f"latest actual older than {cfg.max_actual_age_hours:g} h")
        forecast_age = _age_hours(status.last_issued, now)
        if forecast_age is None:
            reasons.append(REASON_NO_FORECAST)
        elif forecast_age > cfg.max_forecast_age_hours:
            reasons.append(f"latest forecast older than {cfg.max_forecast_age_hours:g} h")
        if status.champion_version is None:
            reasons.append(REASON_NO_CHAMPION)
        elif snap.pointer != state.served.pointer:
            failure = state.reload_failure
            if failure is not None and failure.pointer == snap.pointer:
                reasons.append(failure.reason)
            elif state.reloading:
                reasons.append(REASON_RELOAD_IN_PROGRESS)
            else:
                reasons.append(REASON_OLD_CHAMPION)
    return reasons


# --- App factory ---
def create_app(
    state_loader: Callable[[], AppState] = default_state_loader,
    *,
    config: ServingConfig | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = state_loader()
        if config is not None:
            state.config = config
        app.state.ctx = state
        state.current_snapshot(force=True)
        log.info("champion loaded: %s", state.served.model.version_label)
        yield

    app = FastAPI(
        title="Germany electricity load forecast API",
        version=API_VERSION,
        description=(
            "Day-ahead hourly load forecast for Germany (SMARD region DE) from the champion model."
        ),
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(state: AppState = Depends(get_state)) -> HealthResponse:
        served = state.served
        snap = state.snapshot
        status = None if snap is None else snap.status
        reasons = health_reasons(state, snap)
        now = state.config.now()
        last_actual = None if status is None else status.last_actual
        last_issued = None if status is None else status.last_issued
        return HealthResponse(
            status="degraded" if reasons else "ok",
            reasons=reasons,
            api_version=API_VERSION,
            started_at=state.started_at,
            model=model_info(served.model),
            champion_db_version=None if status is None else status.champion_version,
            database=(
                "unreachable"
                if snap is not None and status is None
                else state.engine.url.get_backend_name()
            ),
            checked_at=None if snap is None else snap.checked_at.to_pydatetime(),
            last_actual_utc=None if last_actual is None else last_actual.to_pydatetime(),
            last_actual_age_hours=_age_hours(last_actual, now),
            last_forecast_issued_utc=None if last_issued is None else last_issued.to_pydatetime(),
            last_forecast_age_hours=_age_hours(last_issued, now),
        )

    @app.get("/forecast", response_model=ForecastResponse)
    def forecast(
        hours: int = Query(DAY_AHEAD_HOURS, ge=1, le=DAY_AHEAD_HOURS),
        state: AppState = Depends(get_state),
    ) -> ForecastResponse:
        model = state.served.model  # one read: the whole request uses this model
        try:
            result = make_day_ahead_forecast(
                state.engine, model, weather_client=state.weather_client, hours=hours
            )
        except NoDataError as err:
            raise HTTPException(status_code=503, detail=str(err)) from err
        return ForecastResponse(
            model=model_info(model),
            horizon=result.horizon,
            issued_at=result.issued_at.to_pydatetime(),
            last_actual_utc=result.last_actual.to_pydatetime(),
            target_start_utc=result.target_start.to_pydatetime(),
            target_end_utc=result.target_end.to_pydatetime(),
            forecast=forecast_points(result.frame["forecast_mw"]),
        )

    @app.get("/forecast/latest", response_model=ForecastResponse)
    def forecast_latest(state: AppState = Depends(get_state)) -> ForecastResponse:
        latest = db.read_latest_model_forecast(state.engine)
        if latest.empty:
            raise HTTPException(status_code=404, detail="no batch forecast stored yet")
        last_actual = db.latest_timestamp(
            state.engine, db.LoadActual, source=SOURCE_SMARD
        ) or pd.Timestamp(0, tz="UTC")
        info = stored_model_info(state, latest)
        return ForecastResponse(
            model=info,
            horizon=info.horizon,
            issued_at=latest["issued_at"].iloc[0].to_pydatetime(),
            last_actual_utc=last_actual.to_pydatetime(),
            target_start_utc=latest["timestamp_utc"].min().to_pydatetime(),
            target_end_utc=latest["timestamp_utc"].max().to_pydatetime(),
            forecast=forecast_points(latest.set_index("timestamp_utc")["forecast_mw"].sort_index()),
        )

    return app


def stored_model_info(state: AppState, issue: pd.DataFrame) -> ModelInfo:
    """The model that issued ``issue`` (one stored issue's rows), not the loaded one.

    Name, version and horizon come from the stored rows; features from that version's
    export (metadata only), or None when the export was pruned. ``alias`` is ``champion``
    only if that version is the DB champion now.
    """
    name = str(issue["model_name"].iloc[0])
    version = str(issue["model_version"].iloc[0])
    export = db.get_model_artifact(state.engine, name, version, with_bundle=False)
    features = None if export is None else list(json.loads(export["features"]))
    snap = state.snapshot
    status = None if snap is None else snap.status
    is_champion = (
        status is not None and name == state.champion_name and status.champion_version == version
    )
    return ModelInfo(
        name=name,
        version=version,
        alias="champion" if is_champion else None,
        horizon=str(issue["horizon"].iloc[0]),
        n_features=None if features is None else len(features),
        features=features,
    )


app = create_app()
