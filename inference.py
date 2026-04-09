"""
Inference Script for SQLGym Environment
===================================
MANDATORY
- Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Uses OpenAI Client for all LLM calls
- Emits structured [START], [STEP], [END] stdout logs

Usage:
    export HF_TOKEN=hf_...
    python inference.py

    # Or with custom settings:
    export API_BASE_URL=http://localhost:11434/v1
    export MODEL_NAME=qwen2.5:7b
    python inference.py
"""

import asyncio
import os
import re
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import SQLGymEnv
from models import SQLAction


def dbg(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.getenv("ENV_URL") or "https://triggeredcode-sql-gym.hf.space"

BENCHMARK = "sql_gym"
MAX_STEPS = 5
TEMPERATURE = 0.1
MAX_TOKENS = 1024
SUCCESS_SCORE_THRESHOLD = 0.6
SCORE_MIN = 0.01
SCORE_MAX = 0.99

WS_CONNECT_TIMEOUT = 120.0
WS_MESSAGE_TIMEOUT = 120.0
WARMUP_MAX_RETRIES = 10
WARMUP_INITIAL_DELAY = 3
WARMUP_MAX_TOTAL_S = 120


def clamp_score(s: float) -> float:
    return max(SCORE_MIN, min(SCORE_MAX, s))

SYSTEM_PROMPT = textwrap.dedent("""
    You are a SQL query optimization expert working with DuckDB.
    Given a slow SQL query, table schemas, EXPLAIN plan, table statistics,
    and column cardinality data, rewrite the query to run faster while
    producing identical results.

    Strategy:
    1. Read the EXPLAIN plan to identify the most expensive operators
    2. Check table statistics and column cardinality to understand data distribution
    3. Apply the optimization that targets the biggest bottleneck first
    4. On subsequent steps, read the EXPLAIN ANALYZE timing to refine further

    Rules:
    - Output ONLY the optimized SQL query, no explanations
    - The query must return the exact same result set (same columns, same rows)
    - Do NOT use DDL/DML (no DROP, DELETE, ALTER, INSERT, UPDATE, CREATE TABLE)
    - Do NOT submit the same query twice (repeat penalty applies)

    DuckDB-specific features to leverage:
    - Window functions: ROW_NUMBER, SUM/AVG OVER, LEAD/LAG, NTILE
    - FILTER clause: COUNT(*) FILTER (WHERE condition) for conditional aggregation
    - QUALIFY: filter on window function results without a subquery
    - CTEs: WITH clauses to materialize shared subqueries once
    - Efficient anti-joins: LEFT JOIN ... IS NULL
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", "\\n")[:200]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def build_prompt(obs) -> str:
    parts = [f"Task ({obs.difficulty}): {obs.task_description}"]
    parts.append(f"\nOriginal (slow) query:\n{obs.original_query}")
    parts.append(f"\nTable schemas:\n{obs.schema_info}")
    parts.append(f"\nTable statistics:\n{obs.table_stats}")
    parts.append(f"\nEXPLAIN plan of original:\n{obs.explain_plan}")
    if obs.indexes:
        parts.append(f"\nAvailable indexes:\n{obs.indexes}")
    if obs.hint:
        parts.append(f"\nHint: {obs.hint}")

    if hasattr(obs, "last_result_preview") and obs.last_result_preview and obs.step_number == 0:
        parts.append(f"\n{obs.last_result_preview}")

    remaining = getattr(obs, "steps_remaining", None)

    if obs.step_number > 0:
        if obs.last_error:
            parts.append(f"\nYour last query had an error: {obs.last_error[:300]}")
        else:
            parts.append(f"\nYour last query was correct. Speedup: {obs.speedup:.2f}x, Score: {obs.current_score:.3f}")
            if obs.last_explain:
                parts.append(f"EXPLAIN ANALYZE of your query:\n{obs.last_explain[:800]}")
        if hasattr(obs, "last_result_preview") and obs.last_result_preview:
            feedback_lines = [l for l in obs.last_result_preview.split("\n") if l.startswith("[TIP]") or l.startswith("---")]
            if feedback_lines:
                parts.append("\n".join(feedback_lines))
        if remaining is not None:
            parts.append(f"\nSteps remaining: {remaining}")
        parts.append("Try a different optimization approach.")

    parts.append("\nOptimized SQL query:")
    return "\n".join(parts)


def extract_sql(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "```" in text:
        blocks = re.findall(r"```(?:sql\n)?(.*?)```", text, re.DOTALL)
        if blocks:
            return blocks[0].strip()
    lines = text.strip().split("\n")
    sql_lines = [l for l in lines if l.strip() and not l.strip().startswith("--") and not l.strip().startswith("#")]
    return "\n".join(sql_lines) if sql_lines else text.strip()


async def warmup_space(url: str) -> bool:
    """Wake up the HF Space with HTTP pings before attempting WebSocket."""
    import httpx
    import time

    start = time.monotonic()
    for attempt in range(1, WARMUP_MAX_RETRIES + 1):
        if time.monotonic() - start > WARMUP_MAX_TOTAL_S:
            dbg(f"Warmup budget exhausted ({WARMUP_MAX_TOTAL_S}s)")
            break
        delay = min(WARMUP_INITIAL_DELAY * (1.3 ** (attempt - 1)), 15)
        try:
            async with httpx.AsyncClient(timeout=20.0) as http:
                resp = await http.get(f"{url}/health")
                if resp.status_code == 200:
                    dbg(f"Space awake (attempt {attempt})")
                    return True
                dbg(f"Warmup {attempt}: HTTP {resp.status_code}")
        except Exception as e:
            dbg(f"Warmup {attempt}: {type(e).__name__}")
        await asyncio.sleep(delay)

    dbg("Space did not wake after warmup")
    return False


async def connect_with_retry(env, max_retries: int = 3) -> bool:
    for attempt in range(1, max_retries + 1):
        try:
            await env.connect()
            dbg(f"WebSocket connected (attempt {attempt})")
            return True
        except Exception as e:
            dbg(f"Connect {attempt}/{max_retries}: {type(e).__name__}: {e}")
            if attempt < max_retries:
                await asyncio.sleep(5 * attempt)
    return False


async def run_task(env, client, task_id, task_difficulty):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    best_score = SCORE_MIN
    success = False

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        if result.done:
            rewards.append(SCORE_MIN)
            steps_taken = 1
            log_step(step=1, action="NOOP", reward=SCORE_MIN, done=True, error="task already done on reset")
        else:
            for step in range(1, MAX_STEPS + 1):
                prompt = build_prompt(obs)
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                    )
                    raw = response.choices[0].message.content or ""
                    query = extract_sql(raw)
                except Exception:
                    query = obs.original_query

                result = await env.step(SQLAction(query=query))
                obs = result.observation
                reward = clamp_score(result.reward or SCORE_MIN)
                done = result.done
                error = obs.last_error if obs.last_error else None

                rewards.append(reward)
                steps_taken = step
                best_score = max(best_score, reward)

                log_step(step=step, action=query, reward=reward, done=done, error=error)

                if done:
                    break

        score = clamp_score(max(best_score, SCORE_MIN))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        err_step = steps_taken + 1
        rewards.append(SCORE_MIN)
        log_step(step=err_step, action="ERROR", reward=SCORE_MIN, done=True, error=str(exc)[:200])
        steps_taken = err_step
        score = SCORE_MIN

    if not rewards:
        rewards = [SCORE_MIN]

    log_end(success=success, steps=max(steps_taken, 1), rewards=rewards)
    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


async def main() -> None:
    if not API_KEY:
        dbg("Error: Set HF_TOKEN or API_KEY environment variable")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    dbg(f"Warming up HF Space at {ENV_URL}...")
    space_ready = await warmup_space(ENV_URL)
    if not space_ready:
        dbg("Proceeding anyway — space may respond to WebSocket")

    env = SQLGymEnv(
        base_url=ENV_URL,
        connect_timeout_s=WS_CONNECT_TIMEOUT,
        message_timeout_s=WS_MESSAGE_TIMEOUT,
    )

    results = []
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}

    try:
        connected = await connect_with_retry(env, max_retries=3)
        if not connected:
            raise ConnectionError(f"Could not connect to {ENV_URL} after retries")

        import httpx
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.get(f"{ENV_URL}/tasks")
            resp.raise_for_status()
            tasks = resp.json()["tasks"]

        tasks.sort(key=lambda t: difficulty_order.get(t.get("difficulty", ""), 99))

        for task_info in tasks:
            tid = task_info["task_id"]
            diff = task_info.get("difficulty", "?")
            result = await run_task(env, client, tid, diff)
            results.append(result)

    except Exception as exc:
        dbg(f"Fatal: {type(exc).__name__}: {exc}")
        if not results:
            log_start(task="connection_error", env=BENCHMARK, model=MODEL_NAME)
            log_step(step=1, action="ERROR", reward=SCORE_MIN, done=True, error=str(exc)[:200])
            log_end(success=False, steps=1, rewards=[SCORE_MIN])
            sys.exit(1)

    finally:
        try:
            await env.close()
        except Exception:
            pass

    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0
    passed = sum(1 for r in results if r["success"])

    dbg(f"SQLGym Results: {avg:.3f} avg, {passed}/{len(results)} passed")
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        dbg(f"  {status} {r['task_id']:30s} score={r['score']:.3f} steps={r['steps']}")


if __name__ == "__main__":
    asyncio.run(main())
