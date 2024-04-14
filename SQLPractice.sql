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

SELECT
    s.name AS superhero_name,
    s.power,
    s.team,
    t.base_location
FROM
    superheroes s
JOIN
    teams t ON s.team = t.team_name
WHERE
    s.power IN (
        SELECT
            power
        FROM
            superheroes
        GROUP BY
            power
        HAVING
            COUNT(*) = 1
    )
ORDER BY
    s.team, s.name;
SELECT
    e.employee_id,
    e.name AS employee_name,
    COALESCE(m.name, 'No Manager') AS manager_name
FROM
    employees e
LEFT JOIN
    employees m ON e.manager_id = m.employee_id;

SELECT
    p1.project_id AS Project1,
    p2.project_id AS Project2
FROM
    projects p1
JOIN
    projects p2 ON p1.start_date <= p2.end_date AND p1.end_date >= p2.start_date
WHERE
    p1.project_id < p2.project_id;
