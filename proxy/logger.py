import sqlite3
import os
import json
import yaml
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "cost_log.db")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "pricing.yaml")

with open(CONFIG_PATH, "r") as f:
    _pricing = yaml.safe_load(f)

_models  = _pricing.get("models", {})
_default = _pricing.get("default", {"input": 3.00, "output": 15.00})


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS llm_calls (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT,
            caller           TEXT,
            model            TEXT,
            input_tokens     INTEGER,
            output_tokens    INTEGER,
            cost_usd         REAL,
            original_model   TEXT,
            routed_model     TEXT,
            complexity_score REAL,
            task_type        TEXT,
            signals          TEXT,
            routing_override TEXT,
            cost_saved       REAL
        )
    """)
    # Add Phase 2 columns to existing database if they don't exist yet
    for col, dtype in [
        ("original_model",   "TEXT"),
        ("routed_model",     "TEXT"),
        ("complexity_score", "REAL"),
        ("task_type",        "TEXT"),
        ("signals",          "TEXT"),
        ("routing_override", "TEXT"),
        ("cost_saved",       "REAL"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE llm_calls ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            pass  # column already exists

    conn.commit()
    conn.close()


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _models.get(model, _default)
    cost  = (input_tokens  / 1_000_000 * rates["input"]) + \
            (output_tokens / 1_000_000 * rates["output"])
    return round(cost, 6)


def log_call(
    caller:        str,
    model:         str,
    input_tokens:  int,
    output_tokens: int,
    # Phase 2 fields — optional so Phase 1 calls still work
    original_model:   str | None = None,
    routed_model:     str | None = None,
    complexity_score: float | None = None,
    task_type:        str | None = None,
    signals:          list | None = None,
    routing_override: str | None = None,
    cost_saved:       float | None = None,
):
    cost = calculate_cost(model, input_tokens, output_tokens)

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO llm_calls (
            timestamp, caller, model,
            input_tokens, output_tokens, cost_usd,
            original_model, routed_model,
            complexity_score, task_type,
            signals, routing_override, cost_saved
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        caller, model,
        input_tokens, output_tokens, cost,
        original_model, routed_model,
        complexity_score, task_type,
        json.dumps(signals) if signals else None,
        routing_override, cost_saved,
    ))
    conn.commit()
    conn.close()

    print(
        f"[LOG] caller={caller} | "
        f"original={original_model} | routed={routed_model} | "
        f"score={complexity_score} | tier={task_type} | "
        f"saved=${cost_saved:.6f}" if cost_saved else
        f"[LOG] caller={caller} | model={model} | "
        f"input={input_tokens} | output={output_tokens} | cost=${cost:.8f}"
    )