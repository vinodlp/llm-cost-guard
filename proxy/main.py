import os
import json
import httpx
import sqlite3
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from proxy.logger import init_db, log_call, DB_PATH
from dashboard.routes import router as dashboard_router
from classifier.ensemble import EnsembleClassifier
from router.router import Router

load_dotenv()

ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL = "https://api.anthropic.com"

# Initialise Phase 2 components once at startup
classifier = EnsembleClassifier(api_key=ANTHROPIC_API_KEY)
router     = Router(config_path="config/pricing.yaml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("[STARTUP] Database initialised")
    print("[STARTUP] Classifier and router ready")
    yield
    print("[SHUTDOWN] Server shutting down")


app = FastAPI(title="LLM Cost Guard", lifespan=lifespan)
app.include_router(dashboard_router)


@app.post("/v1/messages")
async def proxy_messages(request: Request):

    # ── Phase 1: gate ──────────────────────────────
    body   = await request.json()
    caller = request.headers.get("x-caller-id")
    if not caller:
        raise HTTPException(
            status_code=400,
            detail="X-Caller-ID header is required."
        )

    # ── Phase 2: classify + route ──────────────────
    messages      = body.get("messages", [])
    system_prompt = body.get("system", "")
    if isinstance(system_prompt, list):
        system_prompt = " ".join(
            b.get("text", "") for b in system_prompt
            if isinstance(b, dict)
        )

    classification = await classifier.classify(messages, system_prompt)

    tier_header = request.headers.get("x-model-tier")
    decision    = router.route(
        classification=classification,
        requested_model=body.get("model", "claude-haiku-4-5-20251001"),
        caller_id=caller,
        tier_header=tier_header,
    )

    # Swap model in request body
    routed_body = {**body, "model": decision.routed_model}

    # ── Forward to Anthropic ───────────────────────
    headers = {
        "x-api-key":          ANTHROPIC_API_KEY,
        "anthropic-version":  "2023-06-01",
        "content-type":       "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ANTHROPIC_BASE_URL}/v1/messages",
            json=routed_body,
            headers=headers,
            timeout=60.0,
        )

    data = response.json()

    # ── Log with Phase 2 fields ────────────────────
    if "usage" in data:
        usage = data["usage"]
        log_call(
            caller=caller,
            model=decision.routed_model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            original_model=decision.original_model,
            routed_model=decision.routed_model,
            complexity_score=decision.classification.complexity_score,
            task_type=decision.classification.task_type.value,
            signals=decision.classification.signals,
            routing_override=decision.override_reason,
            cost_saved=router.calculate_actual_saving(
                decision.original_model,
                decision.routed_model,
                usage.get("output_tokens", 0),
            ),        )

    return JSONResponse(content=data, status_code=response.status_code)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT caller, original_model, routed_model,
               COUNT(*)                    as calls,
               SUM(input_tokens)           as total_input,
               SUM(output_tokens)          as total_output,
               ROUND(SUM(cost_usd), 6)     as total_cost,
               ROUND(SUM(cost_saved), 6)   as total_saved,
               ROUND(AVG(complexity_score),2) as avg_score
        FROM llm_calls
        GROUP BY caller, original_model, routed_model
    """)
    rows = cursor.fetchall()
    conn.close()

    return {
        "summary": [
            {
                "caller":             r[0],
                "original_model":     r[1],
                "routed_model":       r[2],
                "calls":              r[3],
                "total_input_tokens": r[4],
                "total_output_tokens":r[5],
                "total_cost_usd":     r[6],
                "total_saved_usd":    r[7],
                "avg_complexity":     r[8],
            }
            for r in rows
        ]
    }