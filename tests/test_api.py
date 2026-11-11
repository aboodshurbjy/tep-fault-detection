"""The scoring endpoint: contract, validation, and an alarm with attribution."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest import ONSET, SHIFTED, make_runs, real_data
from tep import config
from tep.api import create_app
from tep.data import load_runs
from tep.scoring import Scorer


@pytest.fixture(scope="module")
def client(tiny_scorer):
    with TestClient(create_app(scorer=tiny_scorer)) as test_client:
        yield test_client


def readings(fault: int = 0, start: int = 0, stop: int = 12, seed: int = 21) -> list[list[float]]:
    return make_runs(1, fault=fault, seed=seed)[config.SENSORS].to_numpy()[start:stop].tolist()


def test_schema_describes_what_score_expects(client):
    body = client.get("/schema").json()
    assert [s["name"] for s in body["sensors"]] == config.SENSORS
    assert body["sensors"][8] == {"name": "xmeas_9", "description": "Reactor temperature", "unit": "Reactor"}
    assert body["min_samples"] == config.WINDOW
    assert body["samples_for_alarm"] == config.WINDOW + config.PERSIST - 1
    assert set(body["thresholds"]) == {"pca", "lightgbm"}


def test_normal_window_does_not_alarm(client):
    response = client.post("/score", json={"readings": readings()})
    assert response.status_code == 200
    body = response.json()
    assert body["n_samples"] == 12
    assert body["alarm"] is False
    for model in ("pca", "lightgbm"):
        entry = body["models"][model]
        assert entry["alarm"] is False and entry["over_threshold"] is False
        assert entry["score"] <= entry["threshold"]
        assert entry["attribution"] is None


def test_faulty_window_alarms_and_says_where_to_look(client):
    window = readings(fault=1, start=ONSET + 8, stop=ONSET + 20)       # entirely after the fault starts
    body = client.post("/score", json={"readings": window, "top_k": 3}).json()
    assert body["alarm"] is True
    for model in ("pca", "lightgbm"):
        entry = body["models"][model]
        assert entry["alarm"] is True and entry["score"] > entry["threshold"]
        attribution = entry["attribution"]
        assert attribution["unit"] == "Reactor"
        assert attribution["check"].startswith("Reactor")
        assert len(attribution["top_sensors"]) <= 3
        assert attribution["top_sensors"][0]["sensor"] in SHIFTED
        assert set(attribution["top_sensors"][0]) == {"sensor", "description", "unit", "share"}
    assert body["models"]["pca"]["attribution"]["statistic"] in {"T2", "SPE"}


def test_alarm_is_undecided_until_enough_samples_for_the_persistence_rule(client):
    window = readings(fault=1, start=ONSET + 8, stop=ONSET + 8 + config.WINDOW)   # one full window only
    entry = client.post("/score", json={"readings": window}).json()["models"]["lightgbm"]
    assert entry["over_threshold"] is True
    assert entry["alarm"] is None


def test_score_matches_the_scorer_used_everywhere_else(client, tiny_scorer):
    window = readings(fault=1, start=ONSET, stop=ONSET + 15)
    body = client.post("/score", json={"readings": window}).json()
    direct = tiny_scorer.score_window(make_runs(1, fault=1, seed=21)[config.SENSORS].to_numpy()[ONSET:ONSET + 15])
    for model in ("pca", "lightgbm"):
        assert body["models"][model]["score"] == pytest.approx(direct[model]["score"])


@pytest.mark.parametrize("payload, fragment", [
    ({"readings": readings(stop=5)}, "at least 10 samples"),
    ({"readings": [[1.0, 2.0, 3.0]] * 12}, "52 values"),
    ({"readings": [row[:-1] for row in readings()]}, "52 values"),
    ({"readings": []}, "52 values"),
])
def test_bad_windows_are_rejected_with_a_reason(client, payload, fragment):
    response = client.post("/score", json=payload)
    assert response.status_code == 422
    assert fragment in str(response.json()["detail"])


@pytest.mark.parametrize("payload", [
    {}, {"readings": "nope"}, {"readings": readings(), "top_k": 0},
    {"readings": [[None] * 52] * 12},
])
def test_malformed_requests_are_rejected(client, payload):
    assert client.post("/score", json=payload).status_code == 422


def test_without_a_trained_model_the_api_says_so(tmp_path):
    with TestClient(create_app(model_dir=tmp_path)) as empty:
        assert empty.get("/health").json()["status"] == "no model"
        response = empty.post("/score", json={"readings": readings()})
        assert response.status_code == 503
        assert "tep.train" in response.json()["detail"]


@real_data
def test_real_model_alarms_on_a_real_reactor_cooling_fault():
    if not (config.MODEL_DIR / "meta.json").exists():
        pytest.skip("no trained model")
    onset = config.ONSET_SAMPLES["testing"]
    run = load_runs("faulty_testing", runs=[7], faults=[4])[config.SENSORS].to_numpy()
    with TestClient(create_app(scorer=Scorer.load())) as real:
        before = real.post("/score", json={"readings": run[onset - 40:onset - 20].tolist()}).json()
        after = real.post("/score", json={"readings": run[onset:onset + 20].tolist()}).json()
    assert before["alarm"] is False
    assert after["alarm"] is True
    assert after["models"]["lightgbm"]["attribution"]["unit"] == "Reactor"
    assert after["models"]["pca"]["attribution"]["unit"] == "Reactor"
