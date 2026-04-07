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

An OpenEnv environment that trains AI agents to optimize slow SQL queries. Agents analyze query plans, table schemas, and execution statistics, then rewrite queries to run faster while producing identical results. Graded on correctness and measured speedup.

## Why SQL Optimization?

SQL query optimization is one of the most impactful real-world skills for any developer or data professional. Slow queries cause cascading failures, cost millions in cloud compute, and block product development. Yet optimization requires deep understanding of execution plans, join strategies, and data distribution — exactly the kind of multi-step reasoning that makes it an ideal RL training ground.

SQLGym bridges this gap: agents receive a slow query, full schema context, and EXPLAIN plans, then iteratively rewrite until they achieve measurable speedup. Every score reflects real execution time improvement, not heuristics.

## Learning Ladder — Curriculum Progression

Tasks are organized as a **skill progression ladder** where patterns learned at each level transfer to harder problems:

**Level 1 — Easy (5 tasks, max 5 steps)**: Single-pattern recognition. Each task has exactly one optimization opportunity and includes a hint.

| ID | Pattern | Transferable Skill |
|----|---------|-------------------|
| `e1_union_to_in` | Replace UNION of disjoint sets with IN | Recognizing redundant operations |
| `e2_redundant_distinct` | Remove DISTINCT on unique columns | Identifying unnecessary work |
| `e3_count_to_exists` | Replace COUNT for existence with EXISTS | Early termination patterns |
| `e4_string_groupby` | Replace string concat GROUP BY with columns | Avoiding expensive expressions |
| `e5_remove_order_by` | Eliminate wasted ORDER BY in subqueries | Understanding query plan flow |

**Level 2 — Medium (5 tasks, max 8 steps)**: Multi-step rewrites requiring structural changes. Skills from Level 1 combine — e.g., recognizing redundant scans (from e1) helps with m4's multi-scan consolidation.

| ID | Pattern | Builds On |
|----|---------|-----------|
| `m1_repeated_subquery` | Correlated subqueries → CTE join | e3 (scan reduction) |
| `m2_scalar_to_window` | Scalar subqueries → window functions | e3, e5 (subquery elimination) |
| `m3_redundant_join` | Pre-aggregate to reduce join cardinality | e2 (removing unnecessary work) |
| `m4_single_scan` | Multiple scans → FILTER/CASE aggregation | e1, e4 (consolidation) |
| `m5_not_in_to_antijoin` | NOT IN → LEFT JOIN / IS NULL | e3 (existence patterns) |

**Level 3 — Hard (5 tasks, max 12 steps)**: Complex analytical queries requiring deep understanding of query semantics and execution plans. Combines multiple Level 1+2 skills.

| ID | Pattern | Requires |
|----|---------|----------|
| `h1_subquery_to_window` | N+1 correlated → window functions | m1, m2 (subquery elimination + windows) |
| `h2_selfjoin_to_lead` | Self-join → LEAD/LAG | m3 (join optimization) + windows |
| `h3_multi_pass_to_single` | 4 passes → single FILTER scan | m4 (consolidation at scale) |
| `h4_correlated_to_filter` | N+1 per-store → single GROUP BY | m1, m4 (correlated elimination + FILTER) |
| `h5_nested_to_cte` | Deeply nested subqueries → CTEs | m1 (CTE refactoring) + all scan patterns |

## How It Works

1. Agent receives: slow SQL query, table schemas, EXPLAIN plan, table statistics, available indexes
2. Agent submits an optimized rewrite (just a SQL string)
3. Environment verifies correctness (order-independent result set comparison) and measures execution time
4. Score = f(correctness, speedup) strictly in (0, 1)
5. Agent can iterate — each step provides updated EXPLAIN plans and score feedback

## Action / Observation Space

**Action** — a single field:

| Field   | Type | Description                         |
|---------|------|-------------------------------------|
| `query` | str  | The optimized SQL query to execute  |

**Observation** — full context for the agent:

| Field                | Type  | Description                              |
|---------------------|-------|------------------------------------------|
| `original_query`    | str   | The slow query to optimize               |
| `schema_info`       | str   | CREATE TABLE statements for all tables   |
| `table_stats`       | str   | Row counts + column cardinality per table|
| `explain_plan`      | str   | EXPLAIN output of the original query     |
| `indexes`           | str   | Available indexes                        |
| `correctness`       | bool  | Whether last submission matched original |
| `speedup`           | float | Execution time ratio (original/optimized)|
| `current_score`     | float | Best score so far, in (0, 1)            |
| `last_error`        | str   | Error message if last query failed       |
| `last_explain`      | str   | EXPLAIN ANALYZE of submitted query (with per-operator timing) |
| `last_result_preview`| str  | First rows + feedback + optimization tips|
| `step_number`       | int   | Current step in episode                  |
| `steps_remaining`   | int   | Steps left before episode ends           |
| `hint`              | str   | Optimization hint (easy tasks only)      |

## Scoring

Score is based on correctness (results must match exactly) and speedup ratio:

| Condition   | Score Range | Signal |
|-------------|-------------|--------|
| Incorrect   | 0.01        | Wrong results — try again |
| Correct, < 1x  | 0.10–0.30  | Query is slower than original |
| Correct, 1x–2x | 0.30–0.60  | Minor improvement |
| Correct, 2x–5x | 0.60–0.99  | Good optimization |
| Correct, >= 5x  | 0.99       | Excellent optimization |

All scores are strictly in (0, 1). Scores are continuous, providing rich gradient signal for RL training.

## Baseline Scores

### Golden Reference (hand-written optimal rewrites)

All 15 golden queries produce correct results. Average score ~0.58 (varies by environment due to timing). Scores range from 0.34 to 0.99 across tasks.

### LLM Baseline (Qwen2.5-72B-Instruct, 5 steps per task)

| Difficulty | Tasks | Avg Score | Range |
|-----------|-------|-----------|-------|
| Easy      | 5     | ~0.55     | 0.34–0.99 |
| Medium    | 5     | ~0.49     | 0.24–0.74 |
| Hard      | 5     | ~0.58     | 0.34–0.99 |

Hard tasks like `h3_multi_pass_to_single` and `h4_correlated_to_filter` can achieve high scores when the agent finds the optimal FILTER aggregation pattern.

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

| Method | Path         | Description                              |
|--------|-------------|------------------------------------------|
| GET    | `/health`   | Health check                             |
| GET    | `/tasks`    | List all tasks with descriptions + skill_tags |
| GET    | `/grader`   | Scoring methodology and rules            |
| GET    | `/curriculum`| Skill progression map and technique bank |
| POST   | `/baseline` | Run golden queries on all tasks          |
| POST   | `/reset`    | Reset environment for a task             |
| WS     | `/ws`       | WebSocket for step/reset/state           |

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

- Score strictly in (0, 1) — correctness is binary, speedup is continuous
- Incorrect results get minimum score (0.01) — no partial credit for wrong answers
- Correct results scored by speedup ratio on a continuous scale
- Result preview with timing feedback after each step guides iterative improvement
- EXPLAIN ANALYZE for submitted queries shows per-operator timing — agents see WHERE bottlenecks are
- Contextual tips (e.g. "try window functions", "use FILTER aggregation") based on current score
- Technique guidance on reset for medium/hard tasks — lists applicable optimization patterns
- Repeat penalty: -0.15 for first duplicate, blocked on 3+ repeats (anti-gaming)
- Multi-step episodes allow agents to learn from feedback and refine

## Design Philosophy

**Approach-agnostic grading** — The environment doesn't check for specific optimization patterns. Any rewrite that produces correct results and runs faster gets proportional reward. CASE WHEN scores as well as FILTER, direct JOINs score as well as CTEs.

**Rich observation for reasoning** — Each observation provides everything a human DBA would have: schema, cardinality statistics (ndistinct + selectivity per column), EXPLAIN plan, and after each submission, EXPLAIN ANALYZE with actual per-operator timing. The agent can see WHERE time is spent, not just WHAT the plan looks like.

**Guided exploration** — Easy tasks include explicit hints. Medium/hard tasks include technique guidance on reset. After each step, contextual tips suggest what to try next based on the current speedup score. This helps agents avoid random exploration and focus on high-yield patterns.

**Anti-gaming** — Repeat query penalties prevent agents from scoring by submitting the same query. The first duplicate costs -0.15, and 3+ repeats are blocked entirely.

**Curriculum transfer** — Skills learned at each level directly apply to harder problems. Window functions from e5 help with m2; FILTER aggregation from m4 is essential for h3 and h4.

## Example Agent Interaction

```
RESET task=m1_repeated_subquery (medium)
  Observation: two correlated subqueries scanning order_items per row
  Schema: customers (20K rows), order_items (500K rows)
  [TECHNIQUES] CTE refactoring, window functions, FILTER aggregation...

STEP query="WITH agg AS (SELECT customer_id, COUNT(*) AS cnt, SUM(qty*price) AS total
             FROM order_items WHERE order_date >= '2024-01-01' GROUP BY customer_id)
             SELECT c.customer_id, c.name, c.city, agg.cnt, agg.total
             FROM customers c LEFT JOIN agg ON c.customer_id = agg.customer_id
             WHERE c.city IN ('New York', 'Chicago', 'LA')
             ORDER BY total DESC NULLS LAST LIMIT 50"
  Correct! Speedup: 2.64x (5.2ms → 2.0ms) | Score improved: 0.010 → 0.683
  EXPLAIN ANALYZE: Hash Join (0.8ms), Sequential Scan (0.6ms), Hash Aggregate (0.4ms)
  [TIP] Good speedup! Try further: push filters earlier, eliminate redundant scans...
  Steps remaining: 7
```

## Data Scale

| Difficulty | Tables | Rows per table | Complexity |
|-----------|--------|----------------|------------|
| Easy      | 1–2    | 10K–200K       | Single-pattern optimization |
| Medium    | 2–3    | 20K–500K       | Multi-table joins, subquery rewrites |
| Hard      | 2–3    | 25K–1M         | Analytical queries, window functions |

DuckDB runs entirely in-memory with no external dependencies.

## Project Structure

```
sql_gym/
├── openenv.yaml           # OpenEnv metadata
├── models.py              # SQLAction, SQLObservation, SQLState (Pydantic)
├── client.py              # WebSocket client
├── inference.py           # LLM inference script (hackathon format)
├── Dockerfile             # Multi-stage container build
├── pyproject.toml         # Dependencies (openenv-core, duckdb, pandas)
└── server/
    ├── app.py             # FastAPI server + /tasks, /grader, /baseline
    ├── grading.py         # Correctness + speedup grading engine
    ├── sql_gym_environment.py  # Core environment (reset/step/state)
    └── tasks/
        ├── registry.py    # Task dataclass + registry
        ├── easy.py        # 5 easy tasks (single-pattern)
        ├── medium.py      # 5 medium tasks (multi-step)
        └── hard.py        # 5 hard tasks (complex analytical)
```
