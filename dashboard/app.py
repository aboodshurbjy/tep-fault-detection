"""Operator dashboard: replay a faulty run as if it were happening now.

    streamlit run dashboard/app.py

The run is revealed one sample at a time. When the detector's alarm rule is
met the page shows which unit operation and which sensors to check first.
An alarm clears once the score has stayed under the threshold for an hour.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from tep import config
from tep.alarms import alarm_episodes
from tep.attribution import FAULT_INFO, SENSOR_INFO
from tep.data import DataNotFound, load_runs
from tep.scoring import ModelNotFound, Scorer

TICK_SECONDS = 0.5
CLEAR_SAMPLES = 20                      # an alarm clears after an hour back under the threshold
MODEL_LABELS = {"lightgbm": "LightGBM on window features", "pca": "PCA (T² and SPE)"}
SCORE_TITLES = {"lightgbm": "Fault probability", "pca": "Distance outside normal limits (1 = limit)"}

# Colours by role: one series hue, a reserved status red, neutral ink for annotations.
SERIES = "#2a78d6"
CRITICAL = "#d03b3b"
NEUTRAL = "#7a7975"

st.set_page_config(page_title="TEP fault detection", layout="wide")


@st.cache_resource(show_spinner="Loading models")
def get_scorer() -> Scorer:
    return Scorer.load()


@st.cache_data(show_spinner="Loading run")
def get_run(fault: int, run: int) -> pd.DataFrame:
    return load_runs("faulty_testing", runs=[run], faults=[fault])


@st.cache_data(show_spinner="Scoring run")
def get_scores(fault: int, run: int, model: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    scorer = get_scorer()
    scores = scorer.score_run(model, get_run(fault, run)[scorer.sensors].to_numpy())
    return scores, alarm_episodes(scorer.alarms(model, scores), CLEAR_SAMPLES)


@st.cache_data(show_spinner=False)
def get_attribution(fault: int, run: int, model: str, alarm_index: int) -> dict:
    scorer = get_scorer()
    values = get_run(fault, run)[scorer.sensors].to_numpy()
    return scorer.explain(model, values[:alarm_index + 1])


def clock(samples: int) -> str:
    minutes = samples * config.SAMPLE_MINUTES
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def hours(samples) -> np.ndarray:
    return np.asarray(samples) * config.SAMPLE_MINUTES / 60


def vertical_markers(points: list[tuple[float, str]], color: str, dashed: bool = False, row: int = 0) -> alt.LayerChart:
    """Labelled vertical rules at the given plant times.

    The layer is always present, even with no points, so the chart keeps
    the same structure while its data grows during a replay.
    """
    data = pd.DataFrame({"hours": pd.Series([h for h, _ in points], dtype=float),
                         "label": pd.Series([label for _, label in points], dtype=str)})
    rule = alt.Chart(data).mark_rule(color=color, strokeWidth=2, strokeDash=[4, 4] if dashed else [1, 0]).encode(
        x="hours:Q", tooltip=["label:N", alt.Tooltip("hours:Q", format=".2f", title="Plant time (h)")])
    text = alt.Chart(data).mark_text(align="left", dx=5, baseline="top", color=NEUTRAL, fontSize=12).encode(
        x="hours:Q", y=alt.value(2 + 16 * row), text="label:N")
    return rule + text


def score_chart(scores: np.ndarray, position: int, threshold: float, model: str,
                alarm_starts: list[int], onset_hour: float | None, total: int) -> alt.LayerChart:
    shown = pd.DataFrame({"hours": hours(np.arange(1, position + 1)), "score": scores[:position]}).dropna()
    x = alt.X("hours:Q", title="Plant time (hours)", scale=alt.Scale(domain=[0, float(hours(total))], nice=False),
              axis=alt.Axis(grid=False))
    if model == "pca":
        top = max(10.0, float(shown["score"].max()) * 4) if len(shown) else 10.0
        shown["score"] = shown["score"].clip(lower=0.05)
        y_scale = alt.Scale(type="log", domain=[0.05, top], clamp=True)
    else:
        y_scale = alt.Scale(domain=[0, 1.2], nice=False)       # headroom above 1.0 for the marker labels
    tooltip = [alt.Tooltip("hours:Q", format=".2f", title="Plant time (h)"),
               alt.Tooltip("score:Q", format=".3f", title=SCORE_TITLES[model])]
    y_axis = alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1.0]) if model == "lightgbm" else alt.Axis()
    line = alt.Chart(shown).mark_line(color=SERIES, strokeWidth=2).encode(
        x=x, y=alt.Y("score:Q", title=SCORE_TITLES[model], scale=y_scale, axis=y_axis))
    hover = alt.Chart(shown).mark_point(size=120, opacity=0).encode(x="hours:Q", y="score:Q", tooltip=tooltip)

    limit = pd.DataFrame({"score": [threshold], "label": [f"Alarm threshold {threshold:.2f}"]})
    limit_rule = alt.Chart(limit).mark_rule(color=NEUTRAL, strokeDash=[6, 4]).encode(y="score:Q")
    limit_text = alt.Chart(limit).mark_text(align="right", dx=-6, dy=-8, color=NEUTRAL, fontSize=12).encode(
        x=alt.value("width"), y="score:Q", text="label:N")

    onset_shown = onset_hour is not None and hours(position) >= onset_hour
    alarm_points = [(float(hours(start + 1)), "Alarm raised" if number == len(alarm_starts) - 1 else "Earlier alarm")
                    for number, start in enumerate(alarm_starts)]
    return alt.layer(
        limit_rule, limit_text, line, hover,
        vertical_markers([(onset_hour, "Fault introduced")] if onset_shown else [], NEUTRAL, dashed=True, row=1),
        vertical_markers(alarm_points, CRITICAL),
    ).properties(height=300)


def contribution_chart(top: list[dict]) -> alt.LayerChart:
    data = pd.DataFrame(top)
    data["label"] = data["description"] + "  ·  " + data["unit"]
    order = alt.EncodingSortField(field="share", order="descending")
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330, ticks=False, domain=False)),
        x=alt.X("share:Q", title="Share of the alarm's evidence", axis=alt.Axis(format="%", grid=False),
                scale=alt.Scale(domain=[0, max(0.5, float(data["share"].max()) * 1.2)])),
        tooltip=[alt.Tooltip("sensor:N", title="Tag"), alt.Tooltip("description:N", title="Sensor"),
                 alt.Tooltip("unit:N", title="Unit"), alt.Tooltip("share:Q", format=".1%", title="Share")])
    bars = base.mark_bar(color=SERIES, size=16, cornerRadiusEnd=4)
    values = base.mark_text(align="left", dx=5, color=NEUTRAL, fontSize=12).encode(text=alt.Text("share:Q", format=".0%"))
    return (bars + values).properties(height=38 * len(data) + 40)


def trend_chart(run: pd.DataFrame, sensor: str, position: int, alarm_index: int) -> alt.LayerChart:
    start = max(0, alarm_index - 80)
    shown = pd.DataFrame({"hours": hours(np.arange(start + 1, position + 1)),
                          "value": run[sensor].to_numpy()[start:position]})
    line = alt.Chart(shown).mark_line(color=SERIES, strokeWidth=2).encode(
        x=alt.X("hours:Q", title="Plant time (hours)", axis=alt.Axis(grid=False)),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)))
    hover = alt.Chart(shown).mark_point(size=120, opacity=0).encode(
        x="hours:Q", y="value:Q",
        tooltip=[alt.Tooltip("hours:Q", format=".2f", title="Plant time (h)"), alt.Tooltip("value:Q", format=".3f")])
    marker = vertical_markers([(float(hours(alarm_index + 1)), "Alarm")], CRITICAL)
    return (line + hover + marker).properties(height=170, title=f"{SENSOR_INFO[sensor][0]} ({sensor})")


# ------------------------------------------------------------------ controls

try:
    scorer = get_scorer()
except ModelNotFound as error:
    st.error(f"{error}")
    st.stop()

state = st.session_state
state.setdefault("position", scorer.window)
state.setdefault("playing", False)



def toggle_play() -> None:
    state.playing = not state.playing


def restart() -> None:
    state.position, state.playing = scorer.window, False


with st.sidebar:
    st.header("Replay")
    fault = st.selectbox("Fault", config.FAULTS, index=3,
                         format_func=lambda f: f"{f}: {FAULT_INFO[f][0]}")
    run_number = st.number_input("Simulation run", min_value=1, max_value=500, value=2, step=1)
    model = st.radio("Detector", list(MODEL_LABELS), format_func=MODEL_LABELS.get)
    speed = st.slider("Speed (samples per tick)", 1, 20, 4,
                      help=f"One sample is {config.SAMPLE_MINUTES} minutes of plant time; a tick is {TICK_SECONDS} s.")
    reveal = st.checkbox("Show when the fault was introduced", value=True,
                         help="Known only because this is a simulation. A real operator would not see it.")
    left, right = st.columns(2)
    left.button("Pause" if state.playing else "Start", type="primary", width="stretch", on_click=toggle_play)
    right.button("Restart", width="stretch", on_click=restart)

selection = (fault, int(run_number), model)
if state.get("selection") != selection:
    state.selection, state.position, state.playing = selection, scorer.window, False

try:
    run = get_run(fault, int(run_number))
except DataNotFound as error:
    st.error(f"{error}")
    st.stop()
scores, episodes = get_scores(fault, int(run_number), model)
total = len(run)
onset = config.ONSET_SAMPLES["testing"]

# -------------------------------------------------------------------- replay

st.title("Tennessee Eastman Process: fault watch")
st.caption(f"Replaying test run {int(run_number)} of fault {fault} ({FAULT_INFO[fault][0]}) with "
           f"{MODEL_LABELS[model]}. The alarm needs {scorer.persist} consecutive samples over the threshold "
           f"and clears after {CLEAR_SAMPLES * config.SAMPLE_MINUTES} minutes back under it.")


@st.fragment(run_every=TICK_SECONDS if state.playing else None)
def live() -> None:
    if state.playing:
        state.position = min(total, state.position + speed)
        if state.position >= total:
            state.playing = False
    position = state.position

    seen = [(start, end) for start, end in episodes if start < position]
    active = seen[-1] if seen and position <= seen[-1][1] else None
    alarm_index = active[0] if active else None
    false_alarm = active is not None and alarm_index < onset
    detected = next((start for start, _ in seen if start >= onset), None)

    status, clock_tile, raised_tile, delay_tile = st.columns(4)
    status.metric("Status", "🔴 ALARM" if active else "🟢 Normal")
    clock_tile.metric("Plant time", clock(position), help="Hours and minutes since the start of the run")
    raised_tile.metric("Alarm raised at", clock(alarm_index + 1) if active else "—")
    if reveal and detected is not None:
        delay_tile.metric("Delay after fault", f"{(detected + 1 - onset) * config.SAMPLE_MINUTES} min")
    elif reveal and position > onset:
        delay_tile.metric("Delay after fault", "not detected yet")
    elif reveal:
        delay_tile.metric("Fault introduced at", "not yet")
    st.progress(position / total)

    onset_hour = float(hours(onset)) if reveal else None
    st.altair_chart(score_chart(scores, position, scorer.thresholds[model], model, [start for start, _ in seen],
                                onset_hour, total), width="stretch")

    if active is None:
        if seen:
            st.info(f"No active alarm. The alarm raised at {clock(seen[-1][0] + 1)} cleared after the score "
                    "stayed under the threshold for an hour.")
        else:
            st.info("No alarm. When the detector alarms, the unit and sensors to check appear here.")
        return

    attribution = get_attribution(fault, int(run_number), model, alarm_index)
    if false_alarm and reveal:
        st.warning("This alarm was raised before the fault was introduced: it is a false alarm.")
    if not attribution["unit"]:
        st.warning("The detector alarmed but no sensor stands out as the cause.")
        return

    advice, chart = st.columns([2, 3])
    with advice:
        st.subheader(f"Check first: {attribution['unit']}")
        st.write(attribution["check"])
        if model == "pca":
            reason = ("the usual relationships between sensors have broken (SPE)"
                      if attribution.get("statistic") == "SPE"
                      else "the plant has moved far from its normal operating point (T²)")
            st.caption(f"Alarm cause: {reason}.")
        table = pd.DataFrame(attribution["top_sensors"])[["sensor", "description", "unit", "share"]]
        table["share"] = (table["share"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(table.rename(columns={"sensor": "Tag", "description": "Sensor", "unit": "Unit", "share": "Share"}),
                     hide_index=True, width="stretch")
    with chart:
        st.subheader("Sensors behind the alarm")
        st.altair_chart(contribution_chart(attribution["top_sensors"]), width="stretch")

    st.subheader("Trends of the top sensors")
    for column, entry in zip(st.columns(3), attribution["top_sensors"][:3]):
        column.altair_chart(trend_chart(run, entry["sensor"], position, alarm_index), width="stretch")


live()
