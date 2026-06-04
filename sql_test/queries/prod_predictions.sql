-- Queries del dashboard "Prod Predictions" (tablas prod_predictions, monitoring_runs, baseline_predictions, training_dataset).
-- $batch se sustituye por un batch real en run_tests.py.

-- @query: variable_batch
SELECT DISTINCT batch FROM prod_predictions ORDER BY batch;

-- @query: monitoring_status
SELECT status AS "Estado", n_rows AS "Filas", has_target::text AS "Tiene target", round(rmse::numeric,2) AS "RMSE", round(rmse_ratio::numeric,2) AS "RMSE ratio", round(r2::numeric,4) AS "R2", round(mape::numeric,2) AS "MAPE %", round(max_psi::numeric,4) AS "Max PSI", round(prediction_psi::numeric,4) AS "Pred PSI", round(schema_violation_pct::numeric,2) AS "Schema viol %" FROM monitoring_runs WHERE batch='$batch' ORDER BY run_time DESC LIMIT 1;

-- @query: hist_age_batch
SELECT age AS "batch" FROM prod_predictions WHERE batch='$batch';

-- @query: hist_age_baseline
SELECT age AS "baseline" FROM training_dataset;

-- @query: hist_bmi_batch
SELECT bmi AS "batch" FROM prod_predictions WHERE batch='$batch';

-- @query: children_proportion
SELECT children::text AS children, sum(CASE WHEN src='batch' THEN pct END) AS batch, sum(CASE WHEN src='baseline' THEN pct END) AS baseline FROM (SELECT children, src, n::float8 / sum(n) OVER (PARTITION BY src) AS pct FROM (SELECT children, 'batch' AS src, count(*) n FROM prod_predictions WHERE batch='$batch' GROUP BY children UNION ALL SELECT children, 'baseline', count(*) FROM training_dataset GROUP BY children) a) b GROUP BY children ORDER BY children;

-- @query: smoker_proportion
SELECT smoker, sum(CASE WHEN src='batch' THEN pct END) AS batch, sum(CASE WHEN src='baseline' THEN pct END) AS baseline FROM (SELECT smoker, src, n::float8 / sum(n) OVER (PARTITION BY src) AS pct FROM (SELECT smoker, 'batch' AS src, count(*) n FROM prod_predictions WHERE batch='$batch' GROUP BY smoker UNION ALL SELECT smoker, 'baseline', count(*) FROM training_dataset GROUP BY smoker) a) b GROUP BY smoker ORDER BY smoker;

-- @query: sex_proportion
SELECT sex, sum(CASE WHEN src='batch' THEN pct END) AS batch, sum(CASE WHEN src='baseline' THEN pct END) AS baseline FROM (SELECT sex, src, n::float8 / sum(n) OVER (PARTITION BY src) AS pct FROM (SELECT sex, 'batch' AS src, count(*) n FROM prod_predictions WHERE batch='$batch' GROUP BY sex UNION ALL SELECT sex, 'baseline', count(*) FROM training_dataset GROUP BY sex) a) b GROUP BY sex ORDER BY sex;

-- @query: region_proportion
SELECT region, sum(CASE WHEN src='batch' THEN pct END) AS batch, sum(CASE WHEN src='baseline' THEN pct END) AS baseline FROM (SELECT region, src, n::float8 / sum(n) OVER (PARTITION BY src) AS pct FROM (SELECT region, 'batch' AS src, count(*) n FROM prod_predictions WHERE batch='$batch' GROUP BY region UNION ALL SELECT region, 'baseline', count(*) FROM training_dataset GROUP BY region) a) b GROUP BY region ORDER BY region;

-- @query: predictions_batch
SELECT predicted_charges AS "batch" FROM prod_predictions WHERE batch='$batch';

-- @query: predictions_baseline
SELECT predicted_charges AS "baseline" FROM baseline_predictions;

-- @query: predictions_summary
SELECT 'batch' AS source, count(*) AS n, avg(predicted_charges) AS mean, percentile_cont(0.5) WITHIN GROUP (ORDER BY predicted_charges) AS median, min(predicted_charges) AS min, max(predicted_charges) AS max FROM prod_predictions WHERE batch='$batch' UNION ALL SELECT 'baseline', count(*), avg(predicted_charges), percentile_cont(0.5) WITHIN GROUP (ORDER BY predicted_charges), min(predicted_charges), max(predicted_charges) FROM baseline_predictions;

-- @query: pred_vs_actual
SELECT actual_charges, predicted_charges AS "predicho" FROM prod_predictions WHERE batch='$batch' AND actual_charges IS NOT NULL;

-- @query: diagonal_reference
SELECT g AS actual_charges, g AS "y=x" FROM generate_series(0, 60000, 2000) g;

-- @query: residuals
SELECT (actual_charges - predicted_charges) AS residual FROM prod_predictions WHERE batch='$batch' AND actual_charges IS NOT NULL;
