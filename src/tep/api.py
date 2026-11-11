"""Scoring API.

    uvicorn tep.api:app

POST /score takes the most recent sensor readings, oldest first, and says
whether either detector would alarm on the latest sample and where to look.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .attribution import SENSOR_INFO
from .scoring import ModelNotFound, Scorer


class ScoreRequest(BaseModel):
    readings: list[list[float]] = Field(
        description="Rows of sensor readings in time order, oldest first. Each row holds one value "
                    "per sensor, in the order given by GET /schema.")
    top_k: int = Field(5, ge=1, le=20, description="How many contributing sensors to return")


def create_app(model_dir: Path | str | None = None, scorer: Scorer | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.scorer = scorer or Scorer.load(model_dir)
            app.state.load_error = None
        except ModelNotFound as error:
            app.state.scorer, app.state.load_error = None, str(error)
        yield

    app = FastAPI(title="TEP fault detection", version="0.1.0", lifespan=lifespan)

    def get_scorer() -> Scorer:
        if app.state.scorer is None:
            raise HTTPException(status_code=503, detail=app.state.load_error)
        return app.state.scorer

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok" if app.state.scorer else "no model", "detail": app.state.load_error}

    @app.get("/schema")
    def schema() -> dict:
        """What /score expects: sensor order, window length and the alarm rule."""
        s = get_scorer()
        return {
            "sensors": [{"name": name, "description": SENSOR_INFO[name][0], "unit": SENSOR_INFO[name][1]}
                        for name in s.sensors],
            "sample_minutes": s.sample_minutes,
            "min_samples": s.min_samples,
            "samples_for_alarm": s.samples_for_alarm,
            "persist_samples": s.persist,
            "thresholds": s.thresholds,
        }

    @app.post("/score")
    def score(request: ScoreRequest) -> dict:
        """Score the latest sample of a window of readings with both detectors."""
        s = get_scorer()
        widths = {len(row) for row in request.readings}
        if widths != {len(s.sensors)}:
            raise HTTPException(status_code=422, detail=(
                f"every row needs {len(s.sensors)} values (see GET /schema); got row lengths {sorted(widths)}"))
        try:
            result = s.score_window(np.asarray(request.readings, dtype=float), request.top_k)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "n_samples": len(request.readings),
            "alarm": any(result[m]["alarm"] for m in result),
            "models": result,
        }

    return app


app = create_app()
