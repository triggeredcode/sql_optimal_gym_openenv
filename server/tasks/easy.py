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


# ── E1: Unnecessary subquery wrapping ─────────────────────────────────────────

register_task(Task(
    task_id="e1_remove_subquery",
    difficulty="easy",
    description=(
        "This query wraps a simple filter in an unnecessary subquery. "
        "Flatten it into a single SELECT for better performance."
    ),
    hint="Remove the inner SELECT and combine all conditions into one WHERE clause.",
    max_steps=5,
    original_query="""
        SELECT * FROM (
            SELECT order_id, customer_id, amount, status, region
            FROM orders
            WHERE status = 'completed'
        ) sub
        WHERE sub.amount > 500 AND sub.region = 'East'
        ORDER BY sub.amount DESC
        LIMIT 100
    """,
    golden_query="""
        SELECT order_id, customer_id, amount, status, region
        FROM orders
        WHERE status = 'completed' AND amount > 500 AND region = 'East'
        ORDER BY amount DESC
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


# ── E3: OR → UNION optimization ──────────────────────────────────────────────

def _seed_products(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            name VARCHAR,
            category VARCHAR,
            price DECIMAL(10,2),
            stock INTEGER,
            rating DECIMAL(3,1),
            created_at DATE
        )
    """)
    conn.execute("""
        INSERT INTO products
        SELECT
            i AS product_id,
            'Product_' || i AS name,
            CASE (i % 6)
                WHEN 0 THEN 'Electronics' WHEN 1 THEN 'Clothing'
                WHEN 2 THEN 'Books' WHEN 3 THEN 'Home'
                WHEN 4 THEN 'Sports' ELSE 'Food'
            END AS category,
            ROUND(5 + random() * 495, 2) AS price,
            CAST(random() * 1000 AS INTEGER) AS stock,
            ROUND(1 + random() * 4, 1) AS rating,
            DATE '2022-01-01' + INTERVAL (i % 1000) DAY AS created_at
        FROM generate_series(1, 200000) t(i)
    """)


register_task(Task(
    task_id="e3_or_to_union",
    difficulty="easy",
    description=(
        "This query uses OR conditions across different columns, preventing "
        "efficient filtering. Rewrite using UNION ALL for better performance."
    ),
    hint="Split the OR conditions into separate SELECTs joined with UNION ALL (and deduplicate if needed).",
    max_steps=5,
    original_query="""
        SELECT product_id, name, category, price
        FROM products
        WHERE category = 'Electronics' OR price > 400 OR rating >= 4.5
        ORDER BY product_id
    """,
    golden_query="""
        SELECT DISTINCT product_id, name, category, price FROM (
            SELECT product_id, name, category, price FROM products WHERE category = 'Electronics'
            UNION ALL
            SELECT product_id, name, category, price FROM products WHERE price > 400
            UNION ALL
            SELECT product_id, name, category, price FROM products WHERE rating >= 4.5
        ) t
        ORDER BY product_id
    """,
    setup_db=_seed_products,
))


# ── E4: Avoid function on indexed column ─────────────────────────────────────

def _seed_employees(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE employees (
            emp_id INTEGER PRIMARY KEY,
            name VARCHAR,
            department VARCHAR,
            salary DECIMAL(10,2),
            hire_date DATE,
            email VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO employees
        SELECT
            i AS emp_id,
            'Employee_' || i AS name,
            CASE (i % 5)
                WHEN 0 THEN 'Engineering' WHEN 1 THEN 'Marketing'
                WHEN 2 THEN 'Sales' WHEN 3 THEN 'HR' ELSE 'Finance'
            END AS department,
            ROUND(40000 + random() * 110000, 2) AS salary,
            DATE '2018-01-01' + INTERVAL (i % 2000) DAY AS hire_date,
            'emp' || i || '@company.com' AS email
        FROM generate_series(1, 100000) t(i)
    """)


register_task(Task(
    task_id="e4_avoid_function_on_column",
    difficulty="easy",
    description=(
        "This query applies UPPER() to the department column in the WHERE "
        "clause, preventing index usage. Rewrite to compare directly "
        "since the data is already consistently cased."
    ),
    hint="Instead of UPPER(department) = 'ENGINEERING', compare against the actual stored value.",
    max_steps=5,
    original_query="""
        SELECT emp_id, name, salary
        FROM employees
        WHERE UPPER(department) = 'ENGINEERING'
        AND salary > 80000
        ORDER BY salary DESC
    """,
    golden_query="""
        SELECT emp_id, name, salary
        FROM employees
        WHERE department = 'Engineering'
        AND salary > 80000
        ORDER BY salary DESC
    """,
    setup_db=_seed_employees,
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
