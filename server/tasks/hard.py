"""Hard tasks — complex analytical queries, multi-step rewrites, real-world patterns."""

import duckdb
from .registry import Task, register_task


def _seed_analytics(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE events (
            event_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            event_type VARCHAR,
            page VARCHAR,
            session_id INTEGER,
            timestamp TIMESTAMP,
            duration_ms INTEGER,
            device VARCHAR,
            country VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO events
        SELECT
            i AS event_id,
            (i % 25000) + 1 AS user_id,
            CASE (i % 5)
                WHEN 0 THEN 'page_view' WHEN 1 THEN 'click'
                WHEN 2 THEN 'scroll' WHEN 3 THEN 'purchase' ELSE 'signup'
            END AS event_type,
            CASE (i % 8)
                WHEN 0 THEN '/home' WHEN 1 THEN '/products' WHEN 2 THEN '/cart'
                WHEN 3 THEN '/checkout' WHEN 4 THEN '/profile' WHEN 5 THEN '/search'
                WHEN 6 THEN '/blog' ELSE '/about'
            END AS page,
            (i / 10) + 1 AS session_id,
            TIMESTAMP '2024-01-01' + INTERVAL (i % 525600) MINUTE AS timestamp,
            CAST(100 + random() * 9900 AS INTEGER) AS duration_ms,
            CASE (i % 3) WHEN 0 THEN 'mobile' WHEN 1 THEN 'desktop' ELSE 'tablet' END AS device,
            CASE (i % 6)
                WHEN 0 THEN 'US' WHEN 1 THEN 'UK' WHEN 2 THEN 'DE'
                WHEN 3 THEN 'FR' WHEN 4 THEN 'JP' ELSE 'BR'
            END AS country
        FROM generate_series(1, 500000) t(i)
    """)

    conn.execute("""
        CREATE TABLE user_profiles (
            user_id INTEGER PRIMARY KEY,
            username VARCHAR,
            tier VARCHAR,
            created_at DATE,
            last_active DATE
        )
    """)
    conn.execute("""
        INSERT INTO user_profiles
        SELECT
            i AS user_id,
            'user_' || i AS username,
            CASE (i % 4) WHEN 0 THEN 'free' WHEN 1 THEN 'basic'
                WHEN 2 THEN 'pro' ELSE 'enterprise' END AS tier,
            DATE '2022-01-01' + INTERVAL (i % 1000) DAY AS created_at,
            DATE '2024-06-01' + INTERVAL (i % 180) DAY AS last_active
        FROM generate_series(1, 25000) t(i)
    """)


# ── H1: N+1 subquery → window function ───────────────────────────────────────

register_task(Task(
    task_id="h1_subquery_to_window",
    difficulty="hard",
    description=(
        "This query uses correlated subqueries to rank purchases and compute "
        "running totals within each user's purchase history. Rewrite using "
        "window functions (ROW_NUMBER, SUM OVER) for better performance."
    ),
    hint=None,
    max_steps=12,
    original_query="""
        SELECT
            e.user_id,
            e.event_type,
            e.timestamp,
            e.duration_ms,
            (SELECT COUNT(*)
             FROM events e2
             WHERE e2.user_id = e.user_id
               AND e2.event_type = 'purchase'
               AND e2.country = 'US'
               AND e2.timestamp <= e.timestamp) AS purchase_rank,
            (SELECT SUM(e3.duration_ms)
             FROM events e3
             WHERE e3.user_id = e.user_id
               AND e3.event_type = 'purchase'
               AND e3.country = 'US'
               AND e3.timestamp <= e.timestamp) AS cumulative_duration
        FROM events e
        WHERE e.event_type = 'purchase'
        AND e.country = 'US'
        ORDER BY e.user_id, e.timestamp
        LIMIT 200
    """,
    golden_query="""
        SELECT
            user_id,
            event_type,
            timestamp,
            duration_ms,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp) AS purchase_rank,
            SUM(duration_ms) OVER (PARTITION BY user_id ORDER BY timestamp) AS cumulative_duration
        FROM events
        WHERE event_type = 'purchase' AND country = 'US'
        ORDER BY user_id, timestamp
        LIMIT 200
    """,
    setup_db=_seed_analytics,
))


# ── H2: Self-join for sequential events → LEAD/LAG ──────────────────────────

register_task(Task(
    task_id="h2_selfjoin_to_lead",
    difficulty="hard",
    description=(
        "This query uses an expensive self-join to find consecutive mobile "
        "events for each user. Rewrite using LEAD() window function over "
        "mobile events to avoid the self-join."
    ),
    hint=None,
    max_steps=12,
    original_query="""
        SELECT
            a.user_id,
            a.page AS current_page,
            b.page AS next_page,
            a.timestamp AS current_time,
            b.timestamp AS next_time,
            b.duration_ms - a.duration_ms AS duration_diff
        FROM events a
        JOIN events b ON a.user_id = b.user_id
            AND b.device = 'mobile'
            AND b.timestamp = (
                SELECT MIN(e.timestamp)
                FROM events e
                WHERE e.user_id = a.user_id
                  AND e.device = 'mobile'
                  AND e.timestamp > a.timestamp
            )
        WHERE a.device = 'mobile'
        ORDER BY a.user_id, a.timestamp
        LIMIT 200
    """,
    golden_query="""
        SELECT
            user_id,
            page AS current_page,
            LEAD(page) OVER (PARTITION BY user_id ORDER BY timestamp) AS next_page,
            timestamp AS current_time,
            LEAD(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp) AS next_time,
            LEAD(duration_ms) OVER (PARTITION BY user_id ORDER BY timestamp) - duration_ms AS duration_diff
        FROM events
        WHERE device = 'mobile'
        QUALIFY next_page IS NOT NULL
        ORDER BY user_id, timestamp
        LIMIT 200
    """,
    setup_db=_seed_analytics,
))


# ── H3: Multiple passes → single scan with CASE/FILTER ──────────────────────

register_task(Task(
    task_id="h3_multi_pass_to_single",
    difficulty="hard",
    description=(
        "This query makes separate subqueries for each metric (one for page "
        "views, one for clicks, one for purchases per country). Combine into "
        "a single scan using conditional aggregation (CASE WHEN or FILTER)."
    ),
    hint=None,
    max_steps=12,
    original_query="""
        SELECT
            country,
            (SELECT COUNT(*) FROM events e2 WHERE e2.country = e.country AND e2.event_type = 'page_view') AS views,
            (SELECT COUNT(*) FROM events e3 WHERE e3.country = e.country AND e3.event_type = 'click') AS clicks,
            (SELECT COUNT(*) FROM events e4 WHERE e4.country = e.country AND e4.event_type = 'purchase') AS purchases,
            (SELECT AVG(duration_ms) FROM events e5 WHERE e5.country = e.country) AS avg_duration
        FROM (SELECT DISTINCT country FROM events) e
        ORDER BY views DESC
    """,
    golden_query="""
        SELECT
            country,
            COUNT(*) FILTER (WHERE event_type = 'page_view') AS views,
            COUNT(*) FILTER (WHERE event_type = 'click') AS clicks,
            COUNT(*) FILTER (WHERE event_type = 'purchase') AS purchases,
            AVG(duration_ms) AS avg_duration
        FROM events
        GROUP BY country
        ORDER BY views DESC
    """,
    setup_db=_seed_analytics,
))


# ── H4: N+1 correlated subqueries → single FILTER aggregation ────────────────

def _seed_sales(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE stores (
            store_id INTEGER PRIMARY KEY,
            store_name VARCHAR,
            region VARCHAR,
            opened_date DATE
        )
    """)
    conn.execute("""
        INSERT INTO stores
        SELECT i, 'Store_' || i,
            CASE (i%4) WHEN 0 THEN 'North' WHEN 1 THEN 'South'
                WHEN 2 THEN 'East' ELSE 'West' END,
            DATE '2015-01-01' + INTERVAL (i*30) DAY
        FROM generate_series(1, 200) t(i)
    """)
    conn.execute("""
        CREATE TABLE sales (
            sale_id INTEGER PRIMARY KEY,
            store_id INTEGER,
            product_id INTEGER,
            category VARCHAR,
            sale_date DATE,
            quantity INTEGER,
            amount DECIMAL(10,2)
        )
    """)
    conn.execute("""
        INSERT INTO sales
        SELECT
            i, (i%200)+1, (i%500)+1,
            CASE (i%5) WHEN 0 THEN 'Electronics' WHEN 1 THEN 'Clothing'
                WHEN 2 THEN 'Food' WHEN 3 THEN 'Home' ELSE 'Sports' END,
            DATE '2023-01-01' + INTERVAL (i%730) DAY,
            1 + (i%10),
            ROUND(5 + random()*95, 2)
        FROM generate_series(1, 1000000) t(i)
    """)


register_task(Task(
    task_id="h4_correlated_to_filter",
    difficulty="hard",
    description=(
        "This query computes per-store revenue broken down by category using "
        "four correlated subqueries — each one scans the entire sales table "
        "per store. Rewrite as a single GROUP BY with conditional FILTER "
        "aggregation to scan sales only once."
    ),
    hint=None,
    max_steps=12,
    original_query="""
        SELECT
            store_id,
            (SELECT COALESCE(SUM(amount),0) FROM sales s WHERE s.store_id = base.store_id AND s.category = 'Electronics') AS electronics_rev,
            (SELECT COALESCE(SUM(amount),0) FROM sales s WHERE s.store_id = base.store_id AND s.category = 'Clothing') AS clothing_rev,
            (SELECT COALESCE(SUM(amount),0) FROM sales s WHERE s.store_id = base.store_id AND s.category = 'Food') AS food_rev,
            (SELECT COUNT(*) FROM sales s WHERE s.store_id = base.store_id) AS total_sales
        FROM (SELECT DISTINCT store_id FROM sales) base
        ORDER BY electronics_rev DESC, store_id
        LIMIT 50
    """,
    golden_query="""
        SELECT
            store_id,
            COALESCE(SUM(amount) FILTER (WHERE category = 'Electronics'), 0) AS electronics_rev,
            COALESCE(SUM(amount) FILTER (WHERE category = 'Clothing'), 0) AS clothing_rev,
            COALESCE(SUM(amount) FILTER (WHERE category = 'Food'), 0) AS food_rev,
            COUNT(*) AS total_sales
        FROM sales
        GROUP BY store_id
        ORDER BY electronics_rev DESC, store_id
        LIMIT 50
    """,
    setup_db=_seed_sales,
))


# ── H5: Nested aggregation → CTE refactor ────────────────────────────────────

register_task(Task(
    task_id="h5_nested_to_cte",
    difficulty="hard",
    description=(
        "This deeply nested query computes user engagement metrics through "
        "multiple layers of subqueries. Refactor into a clear CTE structure "
        "and eliminate redundant scans."
    ),
    hint=None,
    max_steps=12,
    original_query="""
        SELECT
            tier,
            AVG(total_events) AS avg_events,
            AVG(total_duration) AS avg_duration,
            COUNT(*) AS num_users
        FROM (
            SELECT
                up.user_id,
                up.tier,
                (SELECT COUNT(*) FROM events e WHERE e.user_id = up.user_id) AS total_events,
                (SELECT SUM(duration_ms) FROM events e2 WHERE e2.user_id = up.user_id) AS total_duration
            FROM user_profiles up
            WHERE up.tier IN ('pro', 'enterprise')
        ) user_stats
        WHERE total_events > 10
        GROUP BY tier
        ORDER BY avg_events DESC
    """,
    golden_query="""
        WITH user_stats AS (
            SELECT
                user_id,
                COUNT(*) AS total_events,
                SUM(duration_ms) AS total_duration
            FROM events
            GROUP BY user_id
            HAVING COUNT(*) > 10
        )
        SELECT
            up.tier,
            AVG(us.total_events) AS avg_events,
            AVG(us.total_duration) AS avg_duration,
            COUNT(*) AS num_users
        FROM user_profiles up
        JOIN user_stats us ON up.user_id = us.user_id
        WHERE up.tier IN ('pro', 'enterprise')
        GROUP BY up.tier
        ORDER BY avg_events DESC
    """,
    setup_db=_seed_analytics,
))
