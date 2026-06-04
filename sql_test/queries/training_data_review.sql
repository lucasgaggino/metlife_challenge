-- Queries del dashboard "Training data review" (tabla training_dataset).
-- Cada bloque empieza con "-- @query: <nombre>".

-- @query: overview
SELECT count(*) AS "Filas", round(avg(charges)::numeric,2) AS "Avg charges", round(avg(age)::numeric,1) AS "Avg age", round(avg(bmi)::numeric,1) AS "Avg BMI", round(avg((smoker='yes')::int)::numeric,3) AS "% Smokers" FROM training_dataset;

-- @query: hist_charges
SELECT charges FROM training_dataset;

-- @query: hist_age
SELECT age FROM training_dataset;

-- @query: hist_bmi
SELECT bmi FROM training_dataset;

-- @query: count_children
SELECT children::text AS children, count(*) AS count FROM training_dataset GROUP BY children ORDER BY children;

-- @query: count_smoker
SELECT smoker, count(*) AS count FROM training_dataset GROUP BY smoker ORDER BY smoker;

-- @query: count_sex
SELECT sex, count(*) AS count FROM training_dataset GROUP BY sex ORDER BY sex;

-- @query: count_region
SELECT region, count(*) AS count FROM training_dataset GROUP BY region ORDER BY region;

-- @query: corr_with_charges
SELECT 'smoker' AS feature, corr((smoker='yes')::int::float8, charges) AS corr FROM training_dataset UNION ALL SELECT 'age', corr(age, charges) FROM training_dataset UNION ALL SELECT 'bmi', corr(bmi, charges) FROM training_dataset UNION ALL SELECT 'children', corr(children, charges) FROM training_dataset UNION ALL SELECT 'sex_male', corr((sex='male')::int::float8, charges) FROM training_dataset ORDER BY corr DESC;

-- @query: corr_matrix
SELECT 'age' AS variable, corr(age,age) AS age, corr(age,bmi) AS bmi, corr(age,children) AS children, corr(age,(smoker='yes')::int::float8) AS smoker, corr(age,charges) AS charges FROM training_dataset UNION ALL SELECT 'bmi', corr(bmi,age),corr(bmi,bmi),corr(bmi,children),corr(bmi,(smoker='yes')::int::float8),corr(bmi,charges) FROM training_dataset UNION ALL SELECT 'children', corr(children,age),corr(children,bmi),corr(children,children),corr(children,(smoker='yes')::int::float8),corr(children,charges) FROM training_dataset UNION ALL SELECT 'smoker', corr((smoker='yes')::int::float8,age),corr((smoker='yes')::int::float8,bmi),corr((smoker='yes')::int::float8,children),corr((smoker='yes')::int::float8,(smoker='yes')::int::float8),corr((smoker='yes')::int::float8,charges) FROM training_dataset UNION ALL SELECT 'charges', corr(charges,age),corr(charges,bmi),corr(charges,children),corr(charges,(smoker='yes')::int::float8),corr(charges,charges) FROM training_dataset;

-- @query: charges_by_smoker
SELECT smoker, percentile_cont(0.5) WITHIN GROUP (ORDER BY charges) AS median, avg(charges) AS mean FROM training_dataset GROUP BY smoker ORDER BY smoker;

-- @query: charges_by_region
SELECT region, percentile_cont(0.5) WITHIN GROUP (ORDER BY charges) AS median, avg(charges) AS mean FROM training_dataset GROUP BY region ORDER BY region;

-- @query: charges_by_children
SELECT children::text AS children, percentile_cont(0.5) WITHIN GROUP (ORDER BY charges) AS median FROM training_dataset GROUP BY children ORDER BY children;

-- @query: quartiles_smoker
SELECT smoker, count(*) AS n, min(charges) AS min, percentile_cont(0.25) WITHIN GROUP (ORDER BY charges) AS q1, percentile_cont(0.5) WITHIN GROUP (ORDER BY charges) AS median, percentile_cont(0.75) WITHIN GROUP (ORDER BY charges) AS q3, max(charges) AS max FROM training_dataset GROUP BY smoker ORDER BY smoker;

-- @query: quartiles_region
SELECT region, count(*) AS n, min(charges) AS min, percentile_cont(0.25) WITHIN GROUP (ORDER BY charges) AS q1, percentile_cont(0.5) WITHIN GROUP (ORDER BY charges) AS median, percentile_cont(0.75) WITHIN GROUP (ORDER BY charges) AS q3, max(charges) AS max FROM training_dataset GROUP BY region ORDER BY region;

-- @query: scatter_bmi_smoker_yes
SELECT bmi, charges AS "smoker: yes" FROM training_dataset WHERE smoker='yes';

-- @query: scatter_age_smoker_no
SELECT age, charges AS "smoker: no" FROM training_dataset WHERE smoker='no';
