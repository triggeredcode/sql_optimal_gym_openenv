"""
SQLGym Environment Implementation.

Trains AI agents to optimize SQL queries by rewriting them to run faster
while producing identical results. Uses DuckDB as the database engine.
"""

import re
from uuid import uuid4

import duckdb
from openenv.core.env_server.interfaces import Environment

try:
    from ..models import SQLAction, SQLObservation, SQLState
except ImportError:
    from models import SQLAction, SQLObservation, SQLState

from .grading import (
    grade_query, get_explain, get_schema_info, get_table_stats, get_index_info,
    clamp_score, SCORE_MIN,
)
from .tasks import get_task, get_tasks_by_difficulty, list_tasks, TASK_REGISTRY


FORBIDDEN_PATTERNS = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bTRUNCATE\b",
    r"\bALTER\b",
    r"\bINSERT\b",
    r"\bUPDATE\b",
    r"\bCREATE\s+TABLE\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
]


REPEAT_PENALTY_THRESHOLD = 2
REPEAT_PENALTY = 0.15


class SQLGymEnvironment(Environment):
    """SQL optimization environment with correctness + speedup grading."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = SQLState()
        self._task = None
        self._conn: duckdb.DuckDBPyConnection = None
        self._query_history: list = []

    def get_metadata(self):
        return {
            "name": "sql_gym",
            "description": "SQL Query Optimization Environment — trains AI agents to rewrite slow queries for measurable speedup",
            "version": "0.2.0",
            "author": "triggeredcode",
            "tasks": len(TASK_REGISTRY),
            "difficulties": ["easy", "medium", "hard"],
            "scoring": "correctness × speedup, strictly in (0, 1)",
        }

    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs) -> SQLObservation:
        if self._conn:
            self._conn.close()

        self._conn = duckdb.connect(":memory:")
        self._query_history = []

        if task_id and task_id in TASK_REGISTRY:
            self._task = get_task(task_id)
        else:
            difficulty = kwargs.get("difficulty", "easy")
            tasks = get_tasks_by_difficulty(difficulty)
            if not tasks:
                tasks = get_tasks_by_difficulty("easy")
            import random
            rng = random.Random(seed)
            self._task = rng.choice(tasks)

        self._task.setup_db(self._conn)

        schema = get_schema_info(self._conn)
        stats = get_table_stats(self._conn)
        explain = get_explain(self._conn, self._task.original_query)
        indexes = get_index_info(self._conn)

        self._state = SQLState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            max_steps=self._task.max_steps,
            current_step=0,
            best_score=0.0,
            task_completed=False,
        )

        return SQLObservation(
            done=False,
            reward=SCORE_MIN,
            task_description=self._task.description,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            original_query=self._task.original_query.strip(),
            schema_info=schema,
            table_stats=stats,
            explain_plan=explain,
            indexes=indexes,
            last_query="",
            last_result_preview="",
            last_error="",
            last_explain="",
            correctness=False,
            speedup=0.0,
            current_score=SCORE_MIN,
            step_number=0,
            max_steps=self._task.max_steps,
            hint=self._task.hint if self._task.difficulty == "easy" else None,
        )

    def _normalize_query(self, query: str) -> str:
        """Normalize whitespace for duplicate detection."""
        return " ".join(query.split()).lower().strip()

    def step(self, action: SQLAction, timeout_s=None, **kwargs) -> SQLObservation:
        self._state.step_count += 1
        self._state.current_step += 1

        query = action.query.strip()
        normalized = self._normalize_query(query)

        repeat_count = sum(1 for q in self._query_history if q == normalized)
        self._query_history.append(normalized)

        if repeat_count >= REPEAT_PENALTY_THRESHOLD:
            return self._make_observation(
                last_query=query,
                last_error=(
                    f"Duplicate query (submitted {repeat_count + 1} times). "
                    "Try a different optimization approach."
                ),
            )

        safe, msg = self._check_safety(query)
        if not safe:
            return self._make_observation(
                last_query=query,
                last_error=f"Blocked: {msg}",
            )

        try:
            score, correct, speedup, grade_msg = grade_query(
                self._conn, self._task.original_query, query,
            )
        except Exception as e:
            return self._make_observation(
                last_query=query,
                last_error=f"Grading error: {str(e)[:500]}",
            )

        if repeat_count == 1:
            score = max(SCORE_MIN, score - REPEAT_PENALTY)
            grade_msg += f" | Repeat penalty: -{REPEAT_PENALTY:.2f}"

        try:
            preview_rows = self._conn.execute(query).fetchdf().head(10).to_string(index=False)
        except Exception:
            preview_rows = "(could not preview)"

        explain = get_explain(self._conn, query)

        prev_best = self._state.best_score
        self._state.best_score = max(self._state.best_score, score)

        done = score >= 0.95 or self._state.current_step >= self._state.max_steps
        self._state.task_completed = score >= 0.95

        preview_with_feedback = preview_rows[:2000]
        delta = score - prev_best
        direction = "improved" if delta > 0.001 else ("regressed" if delta < -0.001 else "unchanged")
        preview_with_feedback += f"\n--- {grade_msg} | Score {direction}: {prev_best:.3f} → {score:.3f} ---"

        if correct and speedup < 2.0:
            preview_with_feedback += "\n[TIP] Your query is correct but only marginally faster. Look for structural changes: CTEs, window functions, FILTER aggregation, or join elimination."
        elif correct and speedup >= 2.0 and score < 0.90:
            preview_with_feedback += "\n[TIP] Good speedup! Try further: push filters earlier, eliminate redundant scans, or use DuckDB-specific features like QUALIFY."

        clamped_score = clamp_score(score)
        clamped_best = clamp_score(self._state.best_score)

        return SQLObservation(
            done=done,
            reward=clamped_score,
            task_description=self._task.description,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            original_query=self._task.original_query.strip(),
            schema_info=get_schema_info(self._conn),
            table_stats=get_table_stats(self._conn),
            explain_plan=get_explain(self._conn, self._task.original_query),
            indexes=get_index_info(self._conn),
            last_query=query,
            last_result_preview=preview_with_feedback[:3000],
            last_error="" if correct else grade_msg,
            last_explain=explain,
            correctness=correct,
            speedup=speedup,
            current_score=clamped_best,
            step_number=self._state.current_step,
            max_steps=self._state.max_steps,
            hint=self._task.hint if self._task.difficulty == "easy" else None,
        )

    def _make_observation(self, last_query: str, last_error: str) -> SQLObservation:
        done = self._state.current_step >= self._state.max_steps
        return SQLObservation(
            done=done,
            reward=SCORE_MIN,
            task_description=self._task.description,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            original_query=self._task.original_query.strip(),
            schema_info=get_schema_info(self._conn),
            table_stats=get_table_stats(self._conn),
            explain_plan=get_explain(self._conn, self._task.original_query),
            indexes=get_index_info(self._conn),
            last_query=last_query,
            last_result_preview="",
            last_error=last_error,
            last_explain="",
            correctness=False,
            speedup=0.0,
            current_score=clamp_score(self._state.best_score),
            step_number=self._state.current_step,
            max_steps=self._state.max_steps,
            hint=self._task.hint if self._task.difficulty == "easy" else None,
        )

    @property
    def state(self) -> SQLState:
        return self._state

    def _check_safety(self, query: str) -> tuple:
        upper = query.upper()
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, upper, re.IGNORECASE):
                return False, f"Forbidden SQL operation: {pattern}"
        return True, ""

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
