"""
FastAPI application for the SQLGym Environment.

Exposes the SQLGymEnvironment via HTTP and WebSocket endpoints,
plus competition-specific endpoints: /tasks, /grader, /baseline.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

try:
    from ..models import SQLAction, SQLObservation
    from .sql_gym_environment import SQLGymEnvironment
except (ImportError, ModuleNotFoundError):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import SQLAction, SQLObservation
    from server.sql_gym_environment import SQLGymEnvironment

app = create_app(
    SQLGymEnvironment,
    SQLAction,
    SQLObservation,
    env_name="sql_gym",
    max_concurrent_envs=4,
)


from fastapi import FastAPI

try:
    from .tasks import list_tasks as _list_tasks, TASK_REGISTRY
except (ImportError, ModuleNotFoundError):
    from server.tasks import list_tasks as _list_tasks, TASK_REGISTRY

if isinstance(app, FastAPI):

    @app.get("/tasks")
    async def get_tasks():
        return {
            "tasks": _list_tasks(),
            "action_schema": SQLAction.model_json_schema(),
        }

    @app.get("/curriculum")
    async def curriculum():
        """Skill progression map — shows how tasks build on each other."""
        tasks_by_diff = {}
        for t in TASK_REGISTRY.values():
            tasks_by_diff.setdefault(t.difficulty, []).append({
                "task_id": t.task_id,
                "skill_tags": t.skill_tags or [],
                "max_steps": t.max_steps,
                "description": t.description[:120],
            })
        return {
            "curriculum": {
                "philosophy": (
                    "Tasks form a skill ladder: easy tasks teach single patterns, "
                    "medium tasks combine patterns, hard tasks require multi-step "
                    "rewrites using several patterns together."
                ),
                "progression": [
                    {
                        "level": 1,
                        "difficulty": "easy",
                        "focus": "Single-pattern recognition",
                        "tasks": tasks_by_diff.get("easy", []),
                        "core_skills": [
                            "redundant_operation_removal",
                            "predicate_consolidation",
                            "early_termination",
                            "sort_elimination",
                        ],
                    },
                    {
                        "level": 2,
                        "difficulty": "medium",
                        "focus": "Multi-step rewrites, structural changes",
                        "tasks": tasks_by_diff.get("medium", []),
                        "core_skills": [
                            "cte_refactoring",
                            "window_functions",
                            "filter_aggregation",
                            "anti_join",
                            "join_elimination",
                        ],
                        "builds_on": "Level 1 patterns",
                    },
                    {
                        "level": 3,
                        "difficulty": "hard",
                        "focus": "Complex analytical queries, deep plan understanding",
                        "tasks": tasks_by_diff.get("hard", []),
                        "core_skills": [
                            "correlated_subquery_elimination",
                            "window_functions",
                            "filter_aggregation",
                            "cte_refactoring",
                            "self_join_elimination",
                        ],
                        "builds_on": "Level 1 + 2 patterns combined",
                    },
                ],
                "technique_bank": {
                    "scan_reduction": "Combine multiple table scans into one pass",
                    "filter_aggregation": "COUNT/SUM FILTER (WHERE ...) instead of separate subqueries",
                    "window_functions": "ROW_NUMBER, SUM/AVG OVER, LEAD/LAG for row-relative calculations",
                    "cte_refactoring": "WITH clauses to materialize shared subqueries once",
                    "correlated_subquery_elimination": "Replace per-row subqueries with joins or CTEs",
                    "anti_join": "LEFT JOIN ... IS NULL instead of NOT IN/NOT EXISTS",
                    "predicate_consolidation": "Merge multiple WHERE clauses or UNIONs into IN/OR",
                    "qualify": "DuckDB-specific: filter window function results inline",
                },
            },
        }

    @app.get("/grader")
    async def grader_info():
        return {
            "description": "Tasks are graded strictly in (0, 1) based on correctness and speedup.",
            "scoring": {
                "incorrect": "Score 0.01 — results don't match original",
                "correct_slower": "Score 0.10–0.30 — query is slower than original",
                "correct_1x-2x": "Score 0.30–0.60 — minor improvement",
                "correct_2x-5x": "Score 0.60–0.99 — good optimization",
                "correct_5x+": "Score 0.99 — excellent optimization",
            },
            "details": {
                "timing": "Median of 3 runs for each query",
                "comparison": "Order-independent result set comparison",
                "numeric_tolerance": "1e-4 for float comparison",
                "safety": "DROP, DELETE, ALTER, INSERT, UPDATE are blocked",
                "repeat_penalty": "Submitting the same query twice costs -0.15; 3+ repeats are blocked",
                "score_range": "All scores strictly in (0.01, 0.99)",
            },
        }

    @app.post("/baseline")
    async def run_baseline():
        from .grading import clamp_score
        env = SQLGymEnvironment()
        results = []
        for task_id, task in TASK_REGISTRY.items():
            env.reset(task_id=task_id)
            obs = env.step(SQLAction(query=task.golden_query))
            results.append({
                "task_id": task_id,
                "score": clamp_score(obs.reward),
                "correctness": obs.correctness,
                "speedup": obs.speedup,
                "done": obs.done,
            })
        env.close()

        scores = [r["score"] for r in results]
        return {
            "results": results,
            "average_score": sum(scores) / len(scores) if scores else 0,
            "tasks_evaluated": len(results),
        }


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
