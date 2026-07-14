import sqlite3
import json
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from proxy.logger import DB_PATH

router = APIRouter()


def get_stats():
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Overall summary
    cursor.execute("""
        SELECT 
            COUNT(*)                        as total_calls,
            ROUND(SUM(cost_usd), 4)         as total_cost,
            ROUND(SUM(cost_saved), 4)       as total_saved,
            ROUND(AVG(complexity_score), 2) as avg_complexity
        FROM llm_calls
        WHERE routed_model IS NOT NULL
    """)
    summary = cursor.fetchone()

    # Per caller
    cursor.execute("""
        SELECT 
            caller,
            COUNT(*)                        as calls,
            ROUND(SUM(cost_usd), 6)         as cost,
            ROUND(SUM(cost_saved), 6)       as saved,
            ROUND(AVG(complexity_score), 2) as avg_score
        FROM llm_calls
        WHERE routed_model IS NOT NULL
        GROUP BY caller
        ORDER BY saved DESC
    """)
    callers = cursor.fetchall()

    # Tier distribution
    cursor.execute("""
        SELECT routed_model, COUNT(*) as calls
        FROM llm_calls
        WHERE routed_model IS NOT NULL
        GROUP BY routed_model
        ORDER BY calls DESC
    """)
    tiers = cursor.fetchall()

    # Recent calls
    cursor.execute("""
        SELECT 
            timestamp, caller,
            original_model, routed_model,
            complexity_score, task_type,
            signals, cost_saved
        FROM llm_calls
        WHERE routed_model IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    recent = cursor.fetchall()

    conn.close()
    return summary, callers, tiers, recent


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    summary, callers, tiers, recent = get_stats()

    total_calls    = summary[0] or 0
    total_cost     = summary[1] or 0
    total_saved    = summary[2] or 0
    avg_complexity = summary[3] or 0

    # Build caller rows
    caller_rows = ""
    for c in callers:
        caller_rows += f"""
        <tr>
            <td>{c[0]}</td>
            <td>{c[1]}</td>
            <td>${c[2]:.6f}</td>
            <td>${c[3]:.6f}</td>
            <td>{c[4]}</td>
        </tr>"""

    # Build tier rows
    tier_rows = ""
    for t in tiers:
        tier_rows += f"""
        <tr>
            <td>{t[0]}</td>
            <td>{t[1]}</td>
        </tr>"""

    # Build recent rows
    recent_rows = ""
    for r in recent:
        signals = []
        try:
            signals = json.loads(r[6]) if r[6] else []
        except Exception:
            pass
        recent_rows += f"""
        <tr>
            <td>{r[0][:19]}</td>
            <td>{r[1]}</td>
            <td>{r[2]}</td>
            <td>{r[3]}</td>
            <td>{r[4]}</td>
            <td>{r[5]}</td>
            <td>{", ".join(signals[:2])}</td>
            <td>${f"{r[7]:.6f}" if r[7] else "0.000000"}</td>
        </tr>"""

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM Cost Guard Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body      {{ font-family: sans-serif; margin: 40px; background: #f5f5f5; }}
        h1        {{ color: #333; }}
        h2        {{ color: #555; margin-top: 40px; }}
        .cards    {{ display: flex; gap: 20px; margin: 20px 0; }}
        .card     {{ background: white; padding: 20px 30px; border-radius: 8px;
                     border: 1px solid #ddd; min-width: 150px; }}
        .card h3  {{ margin: 0 0 8px; color: #888; font-size: 13px; }}
        .card p   {{ margin: 0; font-size: 28px; font-weight: bold; color: #333; }}
        .saved    {{ color: #2a9d2a; }}
        table     {{ background: white; border-collapse: collapse; width: 100%;
                     border-radius: 8px; overflow: hidden; }}
        th        {{ background: #333; color: white; padding: 10px 14px;
                     text-align: left; font-size: 13px; }}
        td        {{ padding: 10px 14px; border-bottom: 1px solid #eee;
                     font-size: 13px; }}
        tr:hover  {{ background: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>LLM Cost Guard</h1>
    <p>Auto-refreshes every 30 seconds</p>

    <div class="cards">
        <div class="card">
            <h3>Total Requests</h3>
            <p>{total_calls}</p>
        </div>
        <div class="card">
            <h3>Total Cost</h3>
            <p>${total_cost:.4f}</p>
        </div>
        <div class="card">
            <h3>Total Saved</h3>
            <p class="saved">${total_saved:.4f}</p>
        </div>
        <div class="card">
            <h3>Avg Complexity</h3>
            <p>{avg_complexity}</p>
        </div>
    </div>

    <h2>Per Caller</h2>
    <table>
        <tr>
            <th>Caller</th>
            <th>Calls</th>
            <th>Cost</th>
            <th>Saved</th>
            <th>Avg Score</th>
        </tr>
        {caller_rows}
    </table>

    <h2>Routing Distribution</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Requests</th>
        </tr>
        {tier_rows}
    </table>

    <h2>Recent Requests</h2>
    <table>
        <tr>
            <th>Time</th>
            <th>Caller</th>
            <th>Requested</th>
            <th>Routed To</th>
            <th>Score</th>
            <th>Task</th>
            <th>Signals</th>
            <th>Saved</th>
        </tr>
        {recent_rows}
    </table>
</body>
</html>"""

    return HTMLResponse(content=html)