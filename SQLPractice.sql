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

SELECT
    title,
    author,
    magical_level,
    CASE
        WHEN magical_level >= 9 THEN 'Potion of Deep Wisdom'
        WHEN magical_level BETWEEN 7 AND 8 THEN 'Elixir of Enlightenment'
        ELSE 'Moonlight Mead'
    END AS recommended_potion,
    available_on_full_moon
FROM
    books
WHERE
    magical_level >= 7 OR available_on_full_moon = TRUE
ORDER BY
    magical_level DESC, title;

SELECT
    e.name AS enclosure_name,
    e.theme,
    a.name AS animal_name,
    a.species,
    a.active_hours,
    CASE
        WHEN a.active_hours = 'Night' THEN 'Under the Moonlight'
        WHEN a.active_hours = 'Dusk' OR a.active_hours = 'Dawn' THEN 'Twilight Magic'
        ELSE 'Daylight Enchantment'
    END AS showcase_time
FROM
    animals a
JOIN
    enclosures e ON a.enclosure_id = e.enclosure_id
WHERE
    e.theme IN ('Mythical Creatures', 'Enchanted Beasts')
ORDER BY
    e.name, showcase_time;

SELECT
    e.name AS enclosure_name,
    e.theme,
    a.name AS animal_name,
    a.species,
    a.active_hours,
    CASE
        WHEN a.active_hours = 'Night' THEN 'Under the Moonlight'
        WHEN a.active_hours = 'Dusk' OR a.active_hours = 'Dawn' THEN 'Twilight Magic'
        ELSE 'Daylight Enchantment'
    END AS showcase_time
FROM
    animals a
JOIN
    enclosures e ON a.enclosure_id = e.enclosure_id
WHERE
    e.theme IN ('Mythical Creatures', 'Enchanted Beasts')
ORDER BY
    e.name, showcase_time;

SELECT
    p.name AS planet_name,
    p.climate,
    p.distance_from_earth_ly,
    s.name AS fastest_spaceship,
    ROUND(p.distance_from_earth_ly / s.speed_ly_per_hour, 2) AS travel_time_hours
FROM
    planets p
CROSS JOIN
    (SELECT name, speed_ly_per_hour FROM spaceships ORDER BY speed_ly_per_hour DESC LIMIT 1) s
WHERE
    p.climate = 'Temperate'
    AND p.has_unique_attraction = TRUE
    AND p.distance_from_earth_ly <= 50
ORDER BY
    p.distance_from_earth_ly;
