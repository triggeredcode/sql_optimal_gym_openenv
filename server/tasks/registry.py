"""
Task registry for SQLGym.

Each task defines:
  - A slow/unoptimized SQL query
  - The database schema + seed data setup
  - A golden optimized query for baseline
  - Expected speedup range
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import duckdb


@dataclass
class Task:
    task_id: str
    difficulty: str  # easy | medium | hard
    description: str
    hint: Optional[str]
    max_steps: int
    original_query: str
    golden_query: str
    setup_db: Callable[[duckdb.DuckDBPyConnection], None]
    skill_tags: Optional[List[str]] = None


TASK_REGISTRY: Dict[str, Task] = {}


def register_task(task: Task):
    TASK_REGISTRY[task.task_id] = task
    return task


def get_task(task_id: str) -> Task:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]


def get_tasks_by_difficulty(difficulty: str) -> List[Task]:
    return [t for t in TASK_REGISTRY.values() if t.difficulty == difficulty]


def list_tasks() -> List[Dict]:
    return [
        {
            "task_id": t.task_id,
            "difficulty": t.difficulty,
            "description": t.description,
            "max_steps": t.max_steps,
            "skill_tags": t.skill_tags or [],
        }
        for t in TASK_REGISTRY.values()
    ]


from . import easy, medium, hard  # noqa: E402, F401
