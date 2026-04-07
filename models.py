"""
Data models for the SQLGym Environment.

SQLGym trains AI agents to optimize slow SQL queries. Agents rewrite queries
to run faster while producing identical results. Graded on correctness
(result set must match) and speedup ratio.
"""

from typing import List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class SQLAction(Action):
    """An optimized SQL query submitted by the agent."""

    query: str = Field(..., description="The rewritten/optimized SQL query")


class SQLObservation(Observation):
    """Current state of the optimization task."""

    task_description: str = Field(default="", description="What needs to be optimized")
    task_id: str = Field(default="", description="Unique task identifier")
    difficulty: str = Field(default="easy", description="easy | medium | hard")

    original_query: str = Field(default="", description="The slow query to optimize")
    schema_info: str = Field(default="", description="Table schemas, columns, types")
    table_stats: str = Field(default="", description="Row counts, cardinality estimates")
    explain_plan: str = Field(default="", description="EXPLAIN output of the original query")
    indexes: str = Field(default="", description="Available indexes")

    last_query: str = Field(default="", description="Last submitted query")
    last_result_preview: str = Field(default="", description="First rows of last query result")
    last_error: str = Field(default="", description="Error if last query failed")
    last_explain: str = Field(default="", description="EXPLAIN output of last submitted query")

    correctness: bool = Field(default=False, description="Whether last query matches original results")
    speedup: float = Field(default=0.0, description="Execution time ratio: original/optimized")
    current_score: float = Field(default=0.0, description="Combined score (correctness × speedup)")

    step_number: int = Field(default=0, description="Current step in episode")
    max_steps: int = Field(default=8, description="Maximum steps allowed")
    steps_remaining: int = Field(default=8, description="Steps left before episode ends")

    hint: Optional[str] = Field(default=None, description="Hint for easy tasks only")


class SQLState(State):
    """Internal state of a SQLGym episode."""

    task_id: str = Field(default="", description="Current task identifier")
    difficulty: str = Field(default="easy", description="Task difficulty level")
    max_steps: int = Field(default=8, description="Max steps for this task")
    current_step: int = Field(default=0, description="Current step count")
    best_score: float = Field(default=0.0, description="Best score achieved so far")
    task_completed: bool = Field(default=False, description="Whether task is solved")
