"""SQLGym Environment Client.

WebSocket-based client for interacting with the SQLGym server.
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import SQLAction, SQLObservation, SQLState
except (ImportError, ModuleNotFoundError):
    from models import SQLAction, SQLObservation, SQLState


class SQLGymEnv(EnvClient[SQLAction, SQLObservation, SQLState]):
    """
    Client for the SQLGym Environment.

    Example (async):
        >>> async with SQLGymEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task_id="e1_select_star")
        ...     result = await env.step(SQLAction(query="SELECT order_id, amount, status FROM orders WHERE ..."))
        ...     print(f"Score: {result.reward}, Speedup: {result.observation.speedup}x")
    """

    def _step_payload(self, action: SQLAction) -> Dict[str, Any]:
        return {"query": action.query}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SQLObservation]:
        obs_data = payload.get("observation", {})
        observation = SQLObservation(
            done=obs_data.get("done", False),
            reward=payload.get("reward"),
            task_description=obs_data.get("task_description", ""),
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            original_query=obs_data.get("original_query", ""),
            schema_info=obs_data.get("schema_info", ""),
            table_stats=obs_data.get("table_stats", ""),
            explain_plan=obs_data.get("explain_plan", ""),
            indexes=obs_data.get("indexes", ""),
            last_query=obs_data.get("last_query", ""),
            last_result_preview=obs_data.get("last_result_preview", ""),
            last_error=obs_data.get("last_error", ""),
            last_explain=obs_data.get("last_explain", ""),
            correctness=obs_data.get("correctness", False),
            speedup=obs_data.get("speedup", 0.0),
            current_score=obs_data.get("current_score", 0.0),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 8),
            steps_remaining=obs_data.get("steps_remaining", 8),
            hint=obs_data.get("hint"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SQLState:
        return SQLState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            difficulty=payload.get("difficulty", "easy"),
            max_steps=payload.get("max_steps", 8),
            current_step=payload.get("current_step", 0),
            best_score=payload.get("best_score", 0.0),
            task_completed=payload.get("task_completed", False),
        )
