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

    @app.get("/grader")
    async def grader_info():
        return {
            "description": "Tasks are graded 0.0-1.0 based on correctness and speedup.",
            "scoring": {
                "correctness": "Results must exactly match the original query (order-independent)",
                "speedup_5x+": "Score 1.0 — query is 5x or more faster",
                "speedup_2x-5x": "Score 0.6-1.0 — linearly scaled",
                "speedup_1x-2x": "Score 0.3-0.6 — linearly scaled",
                "speedup_<1x": "Score 0.1-0.3 — query is slower than original",
            },
            "details": {
                "timing": "Median of 3 runs for each query",
                "comparison": "Order-independent result set comparison",
                "numeric_tolerance": "1e-4 for float comparison",
                "safety": "DROP, DELETE, ALTER, INSERT, UPDATE are blocked",
            },
        }

    @app.post("/baseline")
    async def run_baseline():
        env = SQLGymEnvironment()
        results = []
        for task_id, task in TASK_REGISTRY.items():
            env.reset(task_id=task_id)
            obs = env.step(SQLAction(query=task.golden_query))
            results.append({
                "task_id": task_id,
                "score": obs.reward,
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
