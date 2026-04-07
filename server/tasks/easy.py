"""Easy tasks — single optimization opportunity, clear improvement path."""

import duckdb
from .registry import Task, register_task


def _seed_orders(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product VARCHAR,
            amount DECIMAL(10,2),
            status VARCHAR,
            order_date DATE,
            region VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO orders
        SELECT
            i AS order_id,
            (i % 500) + 1 AS customer_id,
            CASE (i % 8)
                WHEN 0 THEN 'Laptop' WHEN 1 THEN 'Phone' WHEN 2 THEN 'Tablet'
                WHEN 3 THEN 'Monitor' WHEN 4 THEN 'Keyboard' WHEN 5 THEN 'Mouse'
                WHEN 6 THEN 'Webcam' ELSE 'Headset'
            END AS product,
            ROUND(10 + random() * 990, 2) AS amount,
            CASE (i % 4)
                WHEN 0 THEN 'completed' WHEN 1 THEN 'pending'
                WHEN 2 THEN 'shipped' ELSE 'cancelled'
            END AS status,
            DATE '2023-01-01' + INTERVAL (i % 730) DAY AS order_date,
            CASE (i % 5)
                WHEN 0 THEN 'East' WHEN 1 THEN 'West' WHEN 2 THEN 'North'
                WHEN 3 THEN 'South' ELSE 'Central'
            END AS region
        FROM generate_series(1, 200000) t(i)
    """)


# ── E1: UNION of disjoint sets → IN clause ───────────────────────────────────

register_task(Task(
    task_id="e1_union_to_in",
    difficulty="easy",
    description=(
        "This query uses three separate UNIONs to combine orders by status. "
        "Since each sub-SELECT filters a different status value, the sets are "
        "disjoint — UNION's deduplication is wasted work. Rewrite as a single "
        "scan with an IN clause."
    ),
    hint="The three status filters never overlap, so UNION's dedup adds overhead. Use WHERE status IN (...).",
    max_steps=5,
    skill_tags=["union_elimination", "predicate_consolidation"],
    original_query="""
        SELECT order_id, customer_id, amount, status, region
        FROM (
            SELECT order_id, customer_id, amount, status, region FROM orders WHERE status = 'completed'
            UNION
            SELECT order_id, customer_id, amount, status, region FROM orders WHERE status = 'pending'
            UNION
            SELECT order_id, customer_id, amount, status, region FROM orders WHERE status = 'shipped'
        ) combined
        ORDER BY amount DESC, order_id
        LIMIT 100
    """,
    golden_query="""
        SELECT order_id, customer_id, amount, status, region
        FROM orders
        WHERE status IN ('completed', 'pending', 'shipped')
        ORDER BY amount DESC, order_id
        LIMIT 100
    """,
    setup_db=_seed_orders,
))


# ── E2: Redundant DISTINCT ───────────────────────────────────────────────────

register_task(Task(
    task_id="e2_redundant_distinct",
    difficulty="easy",
    description=(
        "This query uses DISTINCT on a column that already has unique values "
        "(order_id is the primary key). Remove the unnecessary DISTINCT."
    ),
    hint="order_id is already unique — DISTINCT adds overhead for no benefit.",
    max_steps=5,
    skill_tags=["redundant_operation_removal", "key_awareness"],
    original_query="""
        SELECT DISTINCT order_id, customer_id, amount
        FROM orders
        WHERE region = 'East'
        ORDER BY order_id
        LIMIT 200
    """,
    golden_query="""
        SELECT order_id, customer_id, amount
        FROM orders
        WHERE region = 'East'
        ORDER BY order_id
        LIMIT 200
    """,
    setup_db=_seed_orders,
))


# ── E3: COUNT for existence → EXISTS ─────────────────────────────────────────

def _seed_customers_items(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE cust (
            customer_id INTEGER PRIMARY KEY,
            name VARCHAR,
            city VARCHAR,
            signup_date DATE
        )
    """)
    conn.execute("""
        INSERT INTO cust
        SELECT i, 'Customer_' || i,
            CASE (i % 5) WHEN 0 THEN 'NYC' WHEN 1 THEN 'LA' WHEN 2 THEN 'Chicago'
                WHEN 3 THEN 'Houston' ELSE 'Phoenix' END,
            DATE '2020-01-01' + INTERVAL (i % 1500) DAY
        FROM generate_series(1, 10000) t(i)
    """)
    conn.execute("""
        CREATE TABLE items (
            item_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            amount DECIMAL(10,2),
            item_date DATE
        )
    """)
    conn.execute("""
        INSERT INTO items
        SELECT i, (i % 9000) + 1, ROUND(5 + random() * 95, 2),
            DATE '2023-01-01' + INTERVAL (i % 730) DAY
        FROM generate_series(1, 500000) t(i)
    """)


register_task(Task(
    task_id="e3_count_to_exists",
    difficulty="easy",
    description=(
        "This query counts all matching rows just to check if a customer has "
        "ANY orders. COUNT(*) scans every matching row, while EXISTS can stop "
        "at the first match. Replace the COUNT check with EXISTS."
    ),
    hint="EXISTS returns TRUE as soon as it finds one row, avoiding a full count.",
    max_steps=5,
    skill_tags=["early_termination", "existence_check"],
    original_query="""
        SELECT c.customer_id, c.name, c.city
        FROM cust c
        WHERE (SELECT COUNT(*) FROM items i WHERE i.customer_id = c.customer_id) > 0
        ORDER BY c.customer_id
        LIMIT 200
    """,
    golden_query="""
        SELECT c.customer_id, c.name, c.city
        FROM cust c
        WHERE EXISTS (SELECT 1 FROM items i WHERE i.customer_id = c.customer_id)
        ORDER BY c.customer_id
        LIMIT 200
    """,
    setup_db=_seed_customers_items,
))


# ── E4: String concatenation GROUP BY → column GROUP BY ──────────────────────

register_task(Task(
    task_id="e4_string_groupby",
    difficulty="easy",
    description=(
        "This query groups by a concatenated string 'region-status-product'. "
        "String concatenation for grouping is expensive because it allocates "
        "new strings and hashes them. Group by the separate columns instead "
        "and compute the display key in the SELECT."
    ),
    hint="GROUP BY region, status, product is cheaper than GROUP BY region || '-' || status || '-' || product.",
    max_steps=5,
    skill_tags=["expression_pushdown", "groupby_optimization"],
    original_query="""
        SELECT
            region || '-' || status || '-' || product AS group_key,
            COUNT(*) AS cnt,
            SUM(amount) AS total_amount,
            AVG(amount) AS avg_amount,
            MIN(order_date) AS first_order,
            MAX(order_date) AS last_order
        FROM orders
        GROUP BY region || '-' || status || '-' || product
        ORDER BY total_amount DESC
    """,
    golden_query="""
        SELECT
            region || '-' || status || '-' || product AS group_key,
            COUNT(*) AS cnt,
            SUM(amount) AS total_amount,
            AVG(amount) AS avg_amount,
            MIN(order_date) AS first_order,
            MAX(order_date) AS last_order
        FROM orders
        GROUP BY region, status, product
        ORDER BY total_amount DESC
    """,
    setup_db=_seed_orders,
))


# ── E5: Unnecessary ORDER BY in subquery ──────────────────────────────────────

register_task(Task(
    task_id="e5_remove_order_by",
    difficulty="easy",
    description=(
        "This query sorts inside a subquery but the outer query re-sorts "
        "by a different column. The inner ORDER BY is wasted work. "
        "Remove it."
    ),
    hint="ORDER BY in a subquery is discarded when the outer query has its own ORDER BY.",
    max_steps=5,
    skill_tags=["sort_elimination", "subquery_simplification"],
    original_query="""
        SELECT customer_id, total_amount, order_count FROM (
            SELECT
                customer_id,
                SUM(amount) AS total_amount,
                COUNT(*) AS order_count
            FROM orders
            WHERE status = 'completed'
            GROUP BY customer_id
            ORDER BY customer_id
        ) sub
        ORDER BY total_amount DESC
        LIMIT 50
    """,
    golden_query="""
        SELECT
            customer_id,
            SUM(amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE status = 'completed'
        GROUP BY customer_id
        ORDER BY total_amount DESC
        LIMIT 50
    """,
    setup_db=_seed_orders,
))
