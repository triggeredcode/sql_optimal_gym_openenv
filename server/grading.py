"""
SQL query grading engine for SQLGym.

Scores an optimized query on two axes:
  - Correctness: Does the query return the same result set as the original?
  - Speedup: How much faster is the optimized query?

Final score strictly in (0, 1) — never exactly 0.0 or 1.0.
"""

import time
from typing import Tuple

import duckdb


TIMING_RUNS = 3
MAX_QUERY_TIME_S = 10.0
SCORE_MIN = 0.01
SCORE_MAX = 0.99


def clamp_score(score: float) -> float:
    """Clamp score to the open interval (0, 1), required by OpenEnv validator."""
    return max(SCORE_MIN, min(SCORE_MAX, score))


def results_match(
    conn: duckdb.DuckDBPyConnection,
    original_query: str,
    optimized_query: str,
) -> Tuple[bool, str]:
    """Check if two queries produce identical result sets (order-independent)."""
    try:
        orig_result = conn.execute(original_query).fetchdf()
    except Exception as e:
        return False, f"Original query failed: {e}"

    try:
        opt_result = conn.execute(optimized_query).fetchdf()
    except Exception as e:
        return False, f"Optimized query failed: {e}"

    if set(orig_result.columns.tolist()) != set(opt_result.columns.tolist()):
        return False, (
            f"Column mismatch: expected {sorted(orig_result.columns.tolist())}, "
            f"got {sorted(opt_result.columns.tolist())}"
        )

    if len(orig_result) != len(opt_result):
        return False, f"Row count mismatch: expected {len(orig_result)}, got {len(opt_result)}"

    cols = sorted(orig_result.columns.tolist())
    orig_sorted = orig_result[cols].sort_values(by=cols).reset_index(drop=True)
    opt_sorted = opt_result[cols].sort_values(by=cols).reset_index(drop=True)

    for col in cols:
        orig_col = orig_sorted[col]
        opt_col = opt_sorted[col]

        for i in range(len(orig_col)):
            ov = orig_col.iloc[i]
            ev = opt_col.iloc[i]

            if ov is None and ev is None:
                continue
            import pandas as pd
            if pd.isna(ov) and pd.isna(ev):
                continue
            if pd.isna(ov) or pd.isna(ev):
                return False, f"Mismatch at row {i}, col '{col}': {repr(ov)} vs {repr(ev)}"

            if isinstance(ov, float) or isinstance(ev, float):
                try:
                    if abs(float(ov) - float(ev)) > 1e-4:
                        return False, f"Numeric mismatch at row {i}, col '{col}': {ov} vs {ev}"
                except (ValueError, TypeError):
                    if str(ov).strip() != str(ev).strip():
                        return False, f"Mismatch at row {i}, col '{col}': {repr(ov)} vs {repr(ev)}"
            else:
                if str(ov).strip() != str(ev).strip():
                    return False, f"Mismatch at row {i}, col '{col}': {repr(ov)} vs {repr(ev)}"

    return True, ""


def measure_time(conn: duckdb.DuckDBPyConnection, query: str, runs: int = TIMING_RUNS) -> float:
    """Measure median execution time of a query over multiple runs."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        conn.execute(query).fetchall()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    times.sort()
    return times[len(times) // 2]


def grade_query(
    conn: duckdb.DuckDBPyConnection,
    original_query: str,
    optimized_query: str,
) -> Tuple[float, bool, float, str]:
    """
    Grade an optimized query.

    Returns: (score, correct, speedup, message)
    - score: 0.0 to 1.0
    - correct: whether results match
    - speedup: ratio of original_time / optimized_time
    - message: human-readable explanation
    """
    correct, err_msg = results_match(conn, original_query, optimized_query)

    if not correct:
        return SCORE_MIN, False, 0.0, f"Incorrect results: {err_msg}"

    orig_time = measure_time(conn, original_query)
    opt_time = measure_time(conn, optimized_query)

    if opt_time <= 0:
        opt_time = 1e-9

    speedup = orig_time / opt_time

    if speedup >= 5.0:
        speedup_score = 0.99
    elif speedup >= 2.0:
        speedup_score = 0.6 + 0.39 * (speedup - 2.0) / 3.0
    elif speedup >= 1.0:
        speedup_score = 0.3 + 0.3 * (speedup - 1.0)
    else:
        speedup_score = max(0.1, 0.3 * speedup)

    score = clamp_score(speedup_score)

    msg = f"Correct ✓ | Speedup: {speedup:.2f}x ({orig_time*1000:.1f}ms → {opt_time*1000:.1f}ms)"
    return score, True, speedup, msg


def get_explain(conn: duckdb.DuckDBPyConnection, query: str) -> str:
    """Get the EXPLAIN output for a query."""
    try:
        result = conn.execute(f"EXPLAIN {query}").fetchall()
        return "\n".join(str(row[1]) for row in result)
    except Exception as e:
        return f"EXPLAIN failed: {e}"


def get_schema_info(conn: duckdb.DuckDBPyConnection) -> str:
    """Get schema info for all tables in the database."""
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()

    lines = []
    for (table_name,) in tables:
        cols = conn.execute(
            f"SELECT column_name, data_type, is_nullable "
            f"FROM information_schema.columns "
            f"WHERE table_name='{table_name}' ORDER BY ordinal_position"
        ).fetchall()
        col_strs = [f"    {c[0]} {c[1]}{' NOT NULL' if c[2]=='NO' else ''}" for c in cols]
        lines.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(col_strs) + "\n);")
    return "\n\n".join(lines)


def get_table_stats(conn: duckdb.DuckDBPyConnection) -> str:
    """Get row counts and basic stats for all tables."""
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()

    lines = []
    for (table_name,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        lines.append(f"{table_name}: {count:,} rows")
    return "\n".join(lines)


def get_index_info(conn: duckdb.DuckDBPyConnection) -> str:
    """Get index info for the database."""
    try:
        result = conn.execute(
            "SELECT index_name, table_name, is_unique "
            "FROM duckdb_indexes()"
        ).fetchall()
        if not result:
            return "No indexes defined"
        return "\n".join(f"{r[0]} on {r[1]} (unique={r[2]})" for r in result)
    except Exception:
        return "No indexes defined"
