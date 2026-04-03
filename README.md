---
title: SQLGym
emoji: ⚡
colorFrom: purple
colorTo: yellow
sdk: docker
app_port: 8000
tags:
  - openenv
---

# SQLGym — SQL Query Optimization Environment

Train AI agents to optimize slow SQL queries. Agents rewrite queries to run faster while producing identical results, scored on correctness and speedup.

## How It Works

1. Agent receives a slow SQL query, table schemas, EXPLAIN plan, and table statistics
2. Agent submits an optimized rewrite
3. Environment verifies result correctness (order-independent comparison) and measures speedup
4. Score = f(correctness, speedup ratio) in [0.0, 1.0]

## Action / Observation Space

**Action** — a single field:

| Field   | Type | Description                         |
|---------|------|-------------------------------------|
| `query` | str  | The optimized SQL query to execute  |

**Observation** — everything the agent needs:

| Field                | Type  | Description                              |
|---------------------|-------|------------------------------------------|
| `original_query`    | str   | The slow query to optimize               |
| `schema_info`       | str   | CREATE TABLE statements for all tables   |
| `table_stats`       | str   | Row counts per table                     |
| `explain_plan`      | str   | EXPLAIN output of the original query     |
| `indexes`           | str   | Available indexes                        |
| `correctness`       | bool  | Whether last submission matched original |
| `speedup`           | float | Execution time ratio (original/optimized)|
| `current_score`     | float | Best score so far (0.0–1.0)             |
| `last_error`        | str   | Error message if last query failed       |
| `last_explain`      | str   | EXPLAIN of last submitted query          |
| `hint`              | str   | Hint for easy tasks only                 |

## Tasks (15 total)

### Easy (5 tasks, max 5 steps)
| ID | Optimization Pattern |
|----|---------------------|
| `e1_union_to_in` | Replace UNION of disjoint sets with IN clause |
| `e2_redundant_distinct` | Remove DISTINCT on already-unique columns |
| `e3_count_to_exists` | Replace COUNT for existence check with EXISTS |
| `e4_string_groupby` | Replace string concatenation GROUP BY with columns |
| `e5_remove_order_by` | Eliminate wasted ORDER BY in subqueries |

### Medium (5 tasks, max 8 steps)
| ID | Optimization Pattern |
|----|---------------------|
| `m1_repeated_subquery` | Replace repeated correlated subqueries with CTE |
| `m2_scalar_to_window` | Replace scalar subqueries with window functions |
| `m3_redundant_join` | Pre-aggregate to reduce join cardinality |
| `m4_single_scan` | Replace multiple scans with FILTER/CASE aggregation |
| `m5_not_in_to_antijoin` | Rewrite NOT IN as LEFT JOIN / IS NULL |

### Hard (5 tasks, max 12 steps)
| ID | Optimization Pattern |
|----|---------------------|
| `h1_subquery_to_window` | Replace correlated subqueries with window functions |
| `h2_selfjoin_to_lead` | Replace self-join with LEAD/LAG window functions |
| `h3_multi_pass_to_single` | Combine multiple scans into one with FILTER |
| `h4_correlated_to_filter` | Replace N+1 correlated subqueries with FILTER aggregation |
| `h5_nested_to_cte` | Refactor nested subqueries into CTEs |

## Scoring

Score is based on correctness (result sets must match) and speedup:

| Speedup    | Score Range |
|------------|-------------|
| ≥ 5×       | 1.0         |
| 2×–5×      | 0.6–1.0     |
| 1×–2×      | 0.3–0.6     |
| < 1×       | 0.1–0.3     |
| Incorrect  | 0.0         |

## Baseline Scores

### Golden Reference (deterministic SQL rewrites)

All 15 golden queries produce correct results. Average score **0.54** across
all tasks (varies by run due to timing).

### LLM Baseline (qwen2.5:7b, 5 steps per task)

| Difficulty | Tasks | Avg Score | Pass Rate |
|-----------|-------|-----------|-----------|
| Easy      | 5     | 0.628     | 4/5       |
| Medium    | 5     | 0.511     | 4/5       |
| Hard      | 5     | 0.379     | 2/5       |
| **Overall** | **15** | **0.506** | **10/15** |

Notable: h4 (N+1 correlated→FILTER aggregation) scored 1.000 and h1
(correlated→window) scored 0.894, showing 7B models can identify advanced
optimization patterns. Hard tasks like self-join→LEAD and multi-pass→single
remain genuinely challenging.

## Setup

```bash
cd sql_gym
uv sync
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t sql-gym .
docker run -p 8000:8000 sql-gym
```

### Endpoints

| Method | Path       | Description                      |
|--------|-----------|----------------------------------|
| GET    | `/health` | Health check                     |
| GET    | `/tasks`  | List all tasks with descriptions |
| GET    | `/grader` | Scoring methodology              |
| POST   | `/baseline`| Run golden queries on all tasks |
| POST   | `/reset`  | Reset environment for a task     |
| WS     | `/ws`     | WebSocket for step/reset/state   |

### Running Inference

```bash
export HF_TOKEN=hf_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=https://triggeredcode-sql-gym.hf.space
python inference.py
```

## Safety

DDL and DML operations are blocked: `DROP`, `DELETE`, `ALTER`, `INSERT`, `UPDATE`, `CREATE TABLE`, `GRANT`, `REVOKE`.

## Reward Design

- Score = f(correctness, speedup) in [0.0, 1.0] — correctness is binary, speedup is continuous
- Incorrect results always score 0.0 (no partial credit for wrong answers)
- Speedup scoring: ≥5x → 1.0, 2x-5x → 0.6-1.0, 1x-2x → 0.3-0.6, <1x → 0.1-0.3
- **Result preview with feedback** shows timing breakdown and score progress after each step
- Agents see EXPLAIN plans for both original and submitted queries to guide optimization

## Example Agent Interaction

```
RESET task=e1_union_to_in
→ Observation: original query uses UNION of 3 separate SELECTs, schema shows 500K rows

STEP query="SELECT ... WHERE status IN ('completed','pending','shipped') ORDER BY amount DESC, order_id LIMIT 100"
→ Correct ✓ | Speedup: 2.30x (5.1ms → 2.2ms) | Score: 0.641

STEP query="WITH filtered AS (SELECT ... WHERE status IN (...)) SELECT * FROM filtered ORDER BY amount DESC, order_id LIMIT 100"
→ Correct ✓ | Speedup: 2.45x (5.1ms → 2.1ms) | Score: 0.660
```

## Project Structure

```
sql_gym/
├── openenv.yaml           # OpenEnv metadata
├── models.py              # SQLAction, SQLObservation, SQLState
├── client.py              # WebSocket client
├── inference.py           # LLM inference script (hackathon format)
├── Dockerfile             # Container build
├── pyproject.toml         # Dependencies
└── server/
    ├── app.py             # FastAPI server
    ├── grading.py         # Correctness + speedup grading
    ├── sql_gym_environment.py  # Core environment logic
    └── tasks/
        ├── registry.py    # Task registry
        ├── easy.py        # 5 easy tasks (50K–200K rows)
        ├── medium.py      # 5 medium tasks (200K–500K rows)
        └── hard.py        # 5 hard tasks (500K rows)
```
