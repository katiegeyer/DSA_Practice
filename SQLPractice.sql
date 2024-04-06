WITH TopSpenders AS (
  SELECT
    c.customer_id,
    c.customer_name,
    SUM(o.total_amount) AS total_spent
  FROM
    customers c
  JOIN orders o ON c.customer_id = o.customer_id
  WHERE
    o.order_date >= DATEADD(year, -1, GETDATE()) -- Adjust function based on SQL dialect
    AND o.status = 'Completed'
  GROUP BY
    c.customer_id, c.customer_name
  ORDER BY
    total_spent DESC
  LIMIT 3
), PopularCategory AS (
  SELECT
    p.category,
    COUNT(*) AS category_count
  FROM
    order_details od
  JOIN orders o ON od.order_id = o.order_id
  JOIN products p ON od.product_id = p.product_id
  WHERE
    o.customer_id IN (SELECT customer_id FROM TopSpenders)
  GROUP BY
    p.category
  ORDER BY
    category_count DESC
  LIMIT 1
)
SELECT
  ts.customer_name,
  ts.total_spent,
  pc.category AS favorite_category
FROM
  TopSpenders ts,
  PopularCategory pc;
