"""Live dashboard: Germany day-ahead load forecast vs. actuals vs. the official forecast.

Reads straight from the project database (``DATABASE_URL``: Streamlit secret, environment
variable or ``.env``) through the read-only queries in ``dashboard.queries``. Nothing is
computed ahead of time: the batch jobs write forecasts and monitoring events, this page
just scores and draws them.

Run locally with ``make dashboard`` (``streamlit run dashboard/app.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy.engine import Engine

# `streamlit run dashboard/app.py` puts only dashboard/ on the path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings  # noqa: E402
from dashboard import queries as q  # noqa: E402

CACHE_TTL_SECONDS = 10 * 60
COLORS = {
    "actual": "#1f2937",
    "model": "#2563eb",
    "official": "#9ca3af",
    "ok": "#16a34a",
    "bad": "#dc2626",
}
PLOT_TEMPLATE = "plotly_white"
NA = "–"  # noqa: RUF001 - en dash shown for missing values


# --- Data access (cached per database URL) ---
def resolve_database_url() -> str:
    """Streamlit secret first (Community Cloud), then the environment / ``.env`` / default."""
    try:
        secret = st.secrets.get("DATABASE_URL")
    except Exception:  # no secrets.toml at all - normal outside Community Cloud
        secret = None
    return str(secret) if secret else get_settings().database_url


@st.cache_resource(show_spinner=False)
def engine_for(url: str) -> Engine:
    return q.make_engine(url)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_headline(url: str) -> dict[str, Any]:
    eng = engine_for(url)
    return {"champion": q.champion(eng), "coverage": q.coverage(eng)}


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_latest_forecast(url: str) -> pd.DataFrame:
    return q.latest_forecast_frame(engine_for(url))


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_accuracy(url: str, days: int) -> dict[str, Any]:
    eng = engine_for(url)
    as_of = pd.Timestamp.now(tz="UTC").floor("h")
    aligned = q.aligned_frame(eng, days=days, as_of=as_of)
    return {
        "aligned": aligned,
        "daily": q.daily_accuracy(aligned),
        "windows": [q.window_summary(aligned, days=d, as_of=as_of) for d in (7, 30)],
        "official_only": q.official_only_summary(eng, days=30, as_of=as_of),
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_monitoring(url: str) -> dict[str, Any]:
    eng = engine_for(url)
    events = q.monitoring_events(eng)
    return {
        "latest_check": q.latest_check(events),
        "timeline": q.event_timeline(events),
        "versions": q.model_versions(eng),
    }


# --- Formatting helpers ---
def fmt_ts(ts: pd.Timestamp | None, tz: str, fmt: str = "%Y-%m-%d %H:%M") -> str:
    if ts is None or pd.isna(ts):
        return NA
    return pd.Timestamp(ts).tz_convert(tz).strftime(fmt)


def fmt_pct(value: float | None, digits: int = 2) -> str:
    return NA if value is None or pd.isna(value) else f"{value:.{digits}f} %"


def fmt_mw(value: float | None) -> str:
    return NA if value is None or pd.isna(value) else f"{value:,.0f} MW"


def age_text(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return "no data"
    hours = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(ts)) / pd.Timedelta(hours=1)
    if hours < 1.5:
        return "up to date"
    return f"{hours:.0f} h ago" if hours < 48 else f"{hours / 24:.0f} days ago"


def to_tz(series: pd.Series, tz: str) -> pd.Series:
    return series.dt.tz_convert(tz)


# --- Panels ---
def panel_headline(head: dict[str, Any], acc: dict[str, Any], tz: str) -> None:
    champ, cov = head["champion"], head["coverage"]
    w7 = next(w for w in acc["windows"] if w["days"] == 7)
    official30 = acc["official_only"]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if champ is None:
            st.metric("Champion model", "none yet")
            st.caption("No exported model - run the bootstrap workflow.")
        else:
            st.metric("Champion model", f"v{champ['version']}")
            cv = champ["metrics"].get("lightgbm_mape_mean")
            st.caption(
                f"promoted {fmt_ts(champ['promoted_at'], tz, '%Y-%m-%d')} · "
                f"CV MAPE {fmt_pct(cv)} · {champ['n_versions']} version(s)"
            )
    with c2:
        if w7["judged"]:
            gap = w7["model_mape"] - w7["official_mape"]
            st.metric(
                "Model MAPE · last 7 days",
                fmt_pct(w7["model_mape"]),
                delta=f"{gap:+.2f} pp vs official",
                delta_color="inverse",
            )
            st.caption(f"{w7['n_hours']} forecast hours scored against actuals")
        else:
            st.metric("Model MAPE · last 7 days", NA)
            st.caption(
                f"{w7['n_hours']} hour(s) scored so far · needs {q.MIN_WINDOW_HOURS} to judge"
            )
    with c3:
        st.metric("Official forecast MAPE · 30 days", fmt_pct(official30["mape"]))
        st.caption(f"the benchmark to beat · {official30['n_hours']} hours")
    with c4:
        pct = cov["completeness_pct"]
        st.metric("Data coverage", NA if pct is None else f"{pct:.1f} %")
        st.caption(
            f"{cov['n_hours']:,} hourly actuals · latest {fmt_ts(cov['last_actual'], tz)} ({age_text(cov['last_actual'])})"
        )
    with c5:
        first = cov["first_actual"]
        st.metric("Collecting since", fmt_ts(first, tz, "%Y-%m-%d"))
        if first is not None:
            days = (pd.Timestamp.now(tz="UTC") - first).days
            st.caption(
                f"{days:,} days of hourly load · last model forecast {fmt_ts(cov['last_model_forecast'], tz)}"
            )


def panel_forecast(frame: pd.DataFrame, tz: str) -> None:
    st.subheader("Day-ahead forecast vs. actuals")
    if frame.empty:
        st.info("No model forecast in the database yet - the daily forecast job has not run.")
        return
    window = frame[frame["model_mw"].notna()]
    scored = window[window["actual_mw"].notna()]
    issued = window["issued_at"].iloc[0]
    version = window["model_version"].iloc[0]
    x = to_tz(frame["timestamp_utc"], tz)

    fig = go.Figure()
    fig.add_vrect(
        x0=to_tz(window["timestamp_utc"], tz).min(),
        x1=to_tz(window["timestamp_utc"], tz).max(),
        fillcolor="#eff6ff",
        opacity=0.6,
        line_width=0,
        layer="below",
        annotation_text="forecast window",
        annotation_position="top left",
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=frame["official_mw"],
            name="Official day-ahead forecast",
            line={"color": COLORS["official"], "dash": "dot", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=frame["model_mw"],
            name=f"Model forecast (v{version})",
            line={"color": COLORS["model"], "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=frame["actual_mw"],
            name="Actual load",
            line={"color": COLORS["actual"], "width": 2},
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=420,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis_title="MW",
        xaxis_title=f"time ({tz})",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    st.plotly_chart(fig, width="stretch")

    parts = [
        f"Issued {fmt_ts(issued, tz)} · {len(window)} hours · {len(scored)} already have actuals"
    ]
    if len(scored) >= 1:
        model_mae = q.compute_metrics(scored["actual_mw"], scored["model_mw"])["mae"]
        official_mae = q.compute_metrics(scored["actual_mw"], scored["official_mw"])["mae"]
        parts.append(f"so far: model MAE {fmt_mw(model_mae)} · official MAE {fmt_mw(official_mae)}")
    st.caption(" — ".join(parts))


def panel_accuracy(acc: dict[str, Any], head: dict[str, Any], days: int, tz: str) -> None:
    st.subheader("Accuracy over time: model vs. official forecast")
    daily, windows = acc["daily"], acc["windows"]

    cols = st.columns(2)
    for col, w in zip(cols, windows, strict=True):
        with col:
            if not w["judged"]:
                st.markdown(
                    f"**Last {w['days']} days** — {w['n_hours']} hour(s) scored, need {q.MIN_WINDOW_HOURS}"
                )
                continue
            verdict = "model ahead" if w["model_beats_official"] else "official ahead"
            st.markdown(
                f"**Last {w['days']} days** — {w['n_hours']} hours · {verdict} by {abs(w['improvement_pct']):.1f} % on MAE"
            )
            table = pd.DataFrame(
                {
                    "Model": [fmt_pct(w["model_mape"]), fmt_mw(w["model_mae"])],
                    "Official": [fmt_pct(w["official_mape"]), fmt_mw(w["official_mae"])],
                },
                index=["MAPE", "MAE"],
            )
            st.table(table)

    if daily.empty:
        st.info(
            "The live head-to-head starts once forecast hours receive their actuals "
            "(the first full day after the forecast job's first run)."
        )
    else:
        x = daily["date"].dt.tz_localize(None)  # UTC days, drawn as plain dates
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=daily["official_mape"],
                name="Official forecast",
                mode="lines+markers",
                line={"color": COLORS["official"], "width": 2},
                customdata=daily["n_hours"],
                hovertemplate="%{y:.2f} %",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=daily["model_mape"],
                name="Model",
                mode="lines+markers",
                line={"color": COLORS["model"], "width": 3},
                customdata=daily["n_hours"],
                hovertemplate="%{y:.2f} % (%{customdata} h)",
            )
        )
        fig.update_layout(
            template=PLOT_TEMPLATE,
            height=360,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            yaxis_title="daily MAPE (%)",
            xaxis={
                "title": f"UTC day · last {days} days",
                "tickformat": "%b %d",
                "dtick": 86_400_000,
            },
            hovermode="x unified",
            legend={"orientation": "h", "y": 1.1, "x": 0},
        )
        st.plotly_chart(fig, width="stretch")

    champ = head["champion"]
    if champ is not None and champ["metrics"]:
        m = champ["metrics"]
        st.caption(
            f"Offline reference (expanding-window CV on {fmt_ts(champ['train_start'], tz, '%Y-%m')} → "
            f"{fmt_ts(champ['train_end'], tz, '%Y-%m')}): LightGBM MAE {fmt_mw(m.get('lightgbm_mae_mean'))}, "
            f"MAPE {fmt_pct(m.get('lightgbm_mape_mean'))} · seasonal-naive MAE "
            f"{fmt_mw(m.get('seasonal_naive_168_mae_mean'))} (weekly) / {fmt_mw(m.get('seasonal_naive_24_mae_mean'))} (daily)."
        )


def panel_monitoring(mon: dict[str, Any], tz: str) -> None:
    st.subheader("Monitoring status")
    check, timeline, versions = mon["latest_check"], mon["timeline"], mon["versions"]

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Latest check**")
        if check is None:
            st.info("No monitoring events yet - the daily monitor job has not run.")
        else:
            state = "🔴 retraining triggered" if check["triggered"] else "🟢 all triggers clear"
            st.markdown(
                f"{state} · as of {fmt_ts(check['as_of'], tz)} · champion v{check['model_version']}"
            )
            checks = check["checks"]
            if not checks.empty:
                shown = pd.DataFrame(
                    {
                        "trigger": checks["name"],
                        "status": checks["fired"].map({True: "🔴 fired", False: "🟢 ok"}),
                        "value": checks["value"].map(
                            lambda v: NA if v is None or pd.isna(v) else f"{v:.2f}"
                        ),
                        "threshold": checks["threshold"],
                        "detail": checks["detail"],
                    }
                )
                st.dataframe(shown, hide_index=True, width="stretch")
            drift, drift_cols = check["drift"], check["drift_columns"]
            if check["drift_error"]:
                st.warning(f"Drift check failed: {check['drift_error']}")
            elif drift is not None and not drift_cols.empty:
                st.markdown(
                    f"Drift: **{drift['n_drifted']}/{drift['n_monitored']}** features drifted "
                    f"(share {drift['share_drifted']:.0%}, threshold {drift['drift_share_threshold']:.0%}) · "
                    f"prediction drift **{'yes' if drift['prediction_drift'] else 'no'}** · "
                    f"target drift **{'yes' if drift['target_drift'] else 'no'}**"
                )
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=drift_cols["score"],
                        y=drift_cols["column"],
                        orientation="h",
                        name="drift score",
                        marker_color=[
                            COLORS["bad"] if d else COLORS["ok"] for d in drift_cols["drifted"]
                        ],
                        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=drift_cols["threshold"],
                        y=drift_cols["column"],
                        mode="markers",
                        name="threshold",
                        marker={
                            "symbol": "line-ns",
                            "size": 18,
                            "color": COLORS["actual"],
                            "line": {"width": 2},
                        },
                    )
                )
                fig.update_layout(
                    template=PLOT_TEMPLATE,
                    height=max(220, 28 * len(drift_cols) + 80),
                    margin={"l": 10, "r": 10, "t": 10, "b": 10},
                    xaxis_title="normed Wasserstein distance",
                    yaxis={"autorange": "reversed"},
                    showlegend=False,
                )
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "Current window vs. the same season of the training data; the bar is the shift in units of the reference standard deviation, the tick its threshold."
                )

    with right:
        st.markdown("**Model versions**")
        if versions.empty:
            st.info("No exported model versions.")
        else:
            shown = pd.DataFrame(
                {
                    "version": versions["version"].map(lambda v: f"v{v}"),
                    "champion": versions["is_champion"].map({True: "⭐", False: ""}),
                    "trained on": versions.apply(
                        lambda r: (
                            f"{fmt_ts(r['train_start'], tz, '%Y-%m-%d')} → {fmt_ts(r['train_end'], tz, '%Y-%m-%d')}"
                        ),
                        axis=1,
                    ),
                    "CV MAPE": versions["cv_mape"].map(fmt_pct),
                    "exported": versions["created_at"].map(lambda t: fmt_ts(t, tz)),
                    "promoted": versions["promoted_at"].map(lambda t: fmt_ts(t, tz)),
                }
            )
            st.dataframe(shown, hide_index=True, width="stretch")

    st.markdown("**Retraining timeline**")
    if timeline.empty:
        st.caption("No events yet.")
        return
    palette = {
        "ok": COLORS["ok"],
        "triggered": COLORS["bad"],
        "promoted": COLORS["model"],
        "kept": "#f59e0b",
        "skipped": COLORS["official"],
    }
    fig = go.Figure()
    for status, g in timeline.groupby("status", sort=False):
        fig.add_trace(
            go.Scatter(
                x=to_tz(g["created_at"], tz),
                y=g["kind"],
                mode="markers",
                name=status,
                marker={
                    "size": 13,
                    "color": palette.get(status, COLORS["actual"]),
                    "symbol": "diamond" if status == "promoted" else "circle",
                },
                customdata=g[["model_version", "reason"]].fillna("").to_numpy(),
                hovertemplate="%{x}<br>champion v%{customdata[0]}<br>%{customdata[1]}<extra>"
                + status
                + "</extra>",
            )
        )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=220,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title=f"time ({tz})",
        yaxis={"categoryorder": "array", "categoryarray": ["retrain", "check"]},
        legend={"orientation": "h", "y": 1.2, "x": 0},
    )
    st.plotly_chart(fig, width="stretch")
    with st.expander(f"All events ({len(timeline)})"):
        shown = timeline.assign(
            created_at=timeline["created_at"].map(lambda t: fmt_ts(t, tz)),
            as_of=timeline["as_of"].map(lambda t: fmt_ts(t, tz)),
            model_version=timeline["model_version"].map(lambda v: "" if v is None else f"v{v}"),
        )
        st.dataframe(
            shown,
            hide_index=True,
            width="stretch",
            column_config={
                "model_mape_7d": st.column_config.NumberColumn("model MAPE 7d", format="%.2f %%"),
                "official_mape_7d": st.column_config.NumberColumn(
                    "official MAPE 7d", format="%.2f %%"
                ),
                "drift_share": st.column_config.NumberColumn("drift share", format="%.0f %%"),
            },
        )


# --- Page ---
def main() -> None:
    st.set_page_config(page_title="Germany load forecast", page_icon="⚡", layout="wide")
    url = resolve_database_url()

    with st.sidebar:
        st.title("⚡ Germany load forecast")
        st.caption(
            "Day-ahead hourly national load (DE_LU), scored live against the official grid-operator forecast."
        )
        tz = st.radio("Display times in", ["Europe/Berlin", "UTC"], horizontal=True)
        days = st.select_slider(
            "Accuracy window",
            options=[30, 60, 90, 180],
            value=90,
            format_func=lambda d: f"{d} days",
        )
        if st.button("Refresh data", width="stretch"):
            st.cache_data.clear()
        st.caption(f"Source: {q.describe_database(url)} · cached {CACHE_TTL_SECONDS // 60} min")

    # A broken URL or unreachable host should read as a message, not a traceback.
    try:
        head = load_headline(url)
        acc = load_accuracy(url, days)
    except Exception as exc:
        st.error(f"Could not read the database: {exc}")
        st.stop()

    st.title("Germany day-ahead electricity load forecast")
    panel_headline(head, acc, tz)
    st.divider()
    panel_forecast(load_latest_forecast(url), tz)
    st.divider()
    panel_accuracy(acc, head, days, tz)
    st.divider()
    panel_monitoring(load_monitoring(url), tz)


if __name__ == "__main__":  # streamlit runs the script as __main__
    main()
