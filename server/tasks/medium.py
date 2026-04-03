"""Medium tasks — join optimization, subquery refactoring, multi-step improvements."""

import duckdb
from .registry import Task, register_task


def _seed_ecommerce(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name VARCHAR,
            email VARCHAR,
            city VARCHAR,
            signup_date DATE
        )
    """)
    conn.execute("""
        INSERT INTO customers
        SELECT
            i AS customer_id,
            'Customer_' || i AS name,
            'cust' || i || '@mail.com' AS email,
            CASE (i % 8)
                WHEN 0 THEN 'New York' WHEN 1 THEN 'LA' WHEN 2 THEN 'Chicago'
                WHEN 3 THEN 'Houston' WHEN 4 THEN 'Phoenix' WHEN 5 THEN 'Seattle'
                WHEN 6 THEN 'Boston' ELSE 'Denver'
            END AS city,
            DATE '2020-01-01' + INTERVAL (i % 1500) DAY AS signup_date
        FROM generate_series(1, 20000) t(i)
    """)
    conn.execute("""
        CREATE TABLE order_items (
            item_id INTEGER PRIMARY KEY,
            order_id INTEGER,
            customer_id INTEGER,
            product VARCHAR,
            category VARCHAR,
            quantity INTEGER,
            unit_price DECIMAL(10,2),
            order_date DATE
        )
    """)
    conn.execute("""
        INSERT INTO order_items
        SELECT
            i AS item_id,
            (i / 3) + 1 AS order_id,
            (i % 20000) + 1 AS customer_id,
            'Product_' || (i % 200) AS product,
            CASE (i % 6)
                WHEN 0 THEN 'Electronics' WHEN 1 THEN 'Clothing'
                WHEN 2 THEN 'Books' WHEN 3 THEN 'Home'
                WHEN 4 THEN 'Sports' ELSE 'Food'
            END AS category,
            1 + (i % 5) AS quantity,
            ROUND(5 + random() * 195, 2) AS unit_price,
            DATE '2023-01-01' + INTERVAL (i % 730) DAY AS order_date
        FROM generate_series(1, 500000) t(i)
    """)


# ── M1: Repeated subqueries → single scan with window ────────────────────────

register_task(Task(
    task_id="m1_repeated_subquery",
    difficulty="medium",
    description=(
        "This query runs two correlated subqueries per row to find each customer's "
        "order count and total spending. Both scan order_items for the same customer. "
        "Rewrite to scan order_items once using a CTE or subquery join."
    ),
    hint=None,
    max_steps=8,
    original_query="""
        SELECT
            c.customer_id,
            c.name,
            c.city,
            (SELECT COUNT(*)
             FROM order_items oi
             WHERE oi.customer_id = c.customer_id
               AND oi.order_date >= DATE '2024-01-01') AS order_count,
            (SELECT SUM(oi.quantity * oi.unit_price)
             FROM order_items oi
             WHERE oi.customer_id = c.customer_id
               AND oi.order_date >= DATE '2024-01-01') AS total_spent
        FROM customers c
        WHERE c.city IN ('New York', 'Chicago', 'LA')
        ORDER BY total_spent DESC NULLS LAST
        LIMIT 50
    """,
    golden_query="""
        WITH order_agg AS (
            SELECT
                customer_id,
                COUNT(*) AS order_count,
                SUM(quantity * unit_price) AS total_spent
            FROM order_items
            WHERE order_date >= DATE '2024-01-01'
            GROUP BY customer_id
        )
        SELECT
            c.customer_id,
            c.name,
            c.city,
            COALESCE(oa.order_count, 0) AS order_count,
            oa.total_spent
        FROM customers c
        LEFT JOIN order_agg oa ON c.customer_id = oa.customer_id
        WHERE c.city IN ('New York', 'Chicago', 'LA')
        ORDER BY total_spent DESC NULLS LAST
        LIMIT 50
    """,
    setup_db=_seed_ecommerce,
))


# ── M2: Scalar subquery per row → window function ────────────────────────────

register_task(Task(
    task_id="m2_scalar_to_window",
    difficulty="medium",
    description=(
        "This query uses scalar subqueries to compute each order's percentage of "
        "the customer's total spending. Rewrite using window functions to avoid "
        "scanning order_items repeatedly."
    ),
    hint=None,
    max_steps=8,
    original_query="""
        SELECT
            oi.item_id,
            oi.customer_id,
            oi.product,
            oi.quantity * oi.unit_price AS item_total,
            (SELECT SUM(oi2.quantity * oi2.unit_price)
             FROM order_items oi2
             WHERE oi2.customer_id = oi.customer_id
               AND oi2.category = 'Electronics') AS customer_total,
            ROUND(
                (oi.quantity * oi.unit_price) * 100.0 /
                (SELECT SUM(oi3.quantity * oi3.unit_price)
                 FROM order_items oi3
                 WHERE oi3.customer_id = oi.customer_id
                   AND oi3.category = 'Electronics'),
            2) AS pct_of_customer_total
        FROM order_items oi
        WHERE oi.category = 'Electronics'
        ORDER BY pct_of_customer_total DESC, oi.item_id
        LIMIT 100
    """,
    golden_query="""
        SELECT
            item_id,
            customer_id,
            product,
            quantity * unit_price AS item_total,
            SUM(quantity * unit_price) OVER (PARTITION BY customer_id) AS customer_total,
            ROUND(
                (quantity * unit_price) * 100.0 /
                SUM(quantity * unit_price) OVER (PARTITION BY customer_id),
            2) AS pct_of_customer_total
        FROM order_items
        WHERE category = 'Electronics'
        ORDER BY pct_of_customer_total DESC, item_id
        LIMIT 100
    """,
    setup_db=_seed_ecommerce,
))


# ── M3: Redundant joins ──────────────────────────────────────────────────────

def _seed_blog(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE authors (
            author_id INTEGER PRIMARY KEY,
            name VARCHAR,
            bio VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO authors
        SELECT i, 'Author_' || i, 'Bio text for author ' || i
        FROM generate_series(1, 5000) t(i)
    """)
    conn.execute("""
        CREATE TABLE posts (
            post_id INTEGER PRIMARY KEY,
            author_id INTEGER,
            title VARCHAR,
            category VARCHAR,
            published_at DATE,
            views INTEGER
        )
    """)
    conn.execute("""
        INSERT INTO posts
        SELECT
            i AS post_id,
            (i % 5000) + 1 AS author_id,
            'Post Title ' || i AS title,
            CASE (i % 4) WHEN 0 THEN 'Tech' WHEN 1 THEN 'Science'
                WHEN 2 THEN 'Culture' ELSE 'Business' END AS category,
            DATE '2023-01-01' + INTERVAL (i % 730) DAY AS published_at,
            CAST(random() * 10000 AS INTEGER) AS views
        FROM generate_series(1, 200000) t(i)
    """)
    conn.execute("""
        CREATE TABLE comments (
            comment_id INTEGER PRIMARY KEY,
            post_id INTEGER,
            author_id INTEGER,
            body VARCHAR,
            created_at DATE
        )
    """)
    conn.execute("""
        INSERT INTO comments
        SELECT
            i, (i % 200000) + 1, (i % 5000) + 1,
            'Comment body ' || i,
            DATE '2023-06-01' + INTERVAL (i % 500) DAY
        FROM generate_series(1, 500000) t(i)
    """)


register_task(Task(
    task_id="m3_redundant_join",
    difficulty="medium",
    description=(
        "This query joins three tables but only uses columns from two of "
        "them. The join to `authors` is unnecessary since we only need post "
        "data and comment counts. Remove the redundant join."
    ),
    hint=None,
    max_steps=8,
    original_query="""
        SELECT
            p.post_id,
            p.title,
            p.category,
            a.name AS author_name,
            COUNT(c.comment_id) AS comment_count,
            p.views
        FROM posts p
        JOIN authors a ON p.author_id = a.author_id
        LEFT JOIN comments c ON p.post_id = c.post_id
        WHERE p.category = 'Tech'
        GROUP BY p.post_id, p.title, p.category, a.name, p.views
        HAVING COUNT(c.comment_id) > 3
        ORDER BY comment_count DESC, p.post_id
        LIMIT 100
    """,
    golden_query="""
        SELECT
            p.post_id,
            p.title,
            p.category,
            a.name AS author_name,
            sub.comment_count,
            p.views
        FROM posts p
        JOIN authors a ON p.author_id = a.author_id
        JOIN (
            SELECT post_id, COUNT(*) AS comment_count
            FROM comments
            GROUP BY post_id
            HAVING COUNT(*) > 3
        ) sub ON p.post_id = sub.post_id
        WHERE p.category = 'Tech'
        ORDER BY sub.comment_count DESC, p.post_id
        LIMIT 100
    """,
    setup_db=_seed_blog,
))


# ── M4: Multiple scans → single scan with FILTER ─────────────────────────────

register_task(Task(
    task_id="m4_single_scan",
    difficulty="medium",
    description=(
        "This query runs separate subqueries to count orders by category for "
        "each customer. Rewrite to scan order_items once using conditional "
        "aggregation (COUNT FILTER or SUM CASE WHEN)."
    ),
    hint=None,
    max_steps=8,
    original_query="""
        SELECT
            c.customer_id,
            c.name,
            (SELECT COUNT(*) FROM order_items oi
             WHERE oi.customer_id = c.customer_id AND oi.category = 'Electronics') AS electronics_orders,
            (SELECT COUNT(*) FROM order_items oi
             WHERE oi.customer_id = c.customer_id AND oi.category = 'Clothing') AS clothing_orders,
            (SELECT COUNT(*) FROM order_items oi
             WHERE oi.customer_id = c.customer_id AND oi.category = 'Books') AS book_orders,
            (SELECT SUM(oi.quantity * oi.unit_price) FROM order_items oi
             WHERE oi.customer_id = c.customer_id) AS total_spent
        FROM customers c
        ORDER BY total_spent DESC NULLS LAST
        LIMIT 100
    """,
    golden_query="""
        SELECT
            c.customer_id,
            c.name,
            COALESCE(agg.electronics_orders, 0) AS electronics_orders,
            COALESCE(agg.clothing_orders, 0) AS clothing_orders,
            COALESCE(agg.book_orders, 0) AS book_orders,
            agg.total_spent
        FROM customers c
        LEFT JOIN (
            SELECT
                customer_id,
                COUNT(*) FILTER (WHERE category = 'Electronics') AS electronics_orders,
                COUNT(*) FILTER (WHERE category = 'Clothing') AS clothing_orders,
                COUNT(*) FILTER (WHERE category = 'Books') AS book_orders,
                SUM(quantity * unit_price) AS total_spent
            FROM order_items
            GROUP BY customer_id
        ) agg ON c.customer_id = agg.customer_id
        ORDER BY total_spent DESC NULLS LAST
        LIMIT 100
    """,
    setup_db=_seed_ecommerce,
))


# ── M5: NOT IN → anti-join ───────────────────────────────────────────────────

register_task(Task(
    task_id="m5_not_in_to_antijoin",
    difficulty="medium",
    description=(
        "This query uses NOT IN to find customers who have never ordered. "
        "NOT IN has poor performance with large subqueries and NULL-safety "
        "issues. Rewrite as a LEFT JOIN / IS NULL anti-join."
    ),
    hint=None,
    max_steps=8,
    original_query="""
        SELECT customer_id, name, email, city
        FROM customers
        WHERE customer_id NOT IN (
            SELECT DISTINCT customer_id FROM order_items
        )
        ORDER BY customer_id
    """,
    golden_query="""
        SELECT c.customer_id, c.name, c.email, c.city
        FROM customers c
        LEFT JOIN (
            SELECT DISTINCT customer_id FROM order_items
        ) oi ON c.customer_id = oi.customer_id
        WHERE oi.customer_id IS NULL
        ORDER BY c.customer_id
    """,
    setup_db=_seed_ecommerce,
))
