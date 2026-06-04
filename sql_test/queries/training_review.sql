-- Queries del dashboard "Training review" (tablas training_runs y training_feature_importance).
-- $run se sustituye por un run_name real en run_tests.py.

-- @query: variable_run
SELECT run_name FROM training_runs ORDER BY started_at DESC;

-- @query: run_summary
SELECT round(duration_seconds::numeric,1) AS "Duracion (s)", dataset_name AS "Dataset", dataset_rows AS "Filas", round(val_rmse::numeric,2) AS "Val RMSE", round(val_r2::numeric,4) AS "Val R2", round(val_mae::numeric,2) AS "Val MAE", round(val_mape::numeric,2) AS "Val MAPE %", round(overfitting_score::numeric,4) AS "Overfitting", is_champion::text AS "Champion" FROM training_runs WHERE run_name = '$run';

-- @query: error_train_vs_val
SELECT 'RMSE' AS metric, train_rmse AS train, val_rmse AS val FROM training_runs WHERE run_name='$run' UNION ALL SELECT 'MAE', train_mae, val_mae FROM training_runs WHERE run_name='$run' UNION ALL SELECT 'MAPE', train_mape, val_mape FROM training_runs WHERE run_name='$run';

-- @query: r2_train_vs_val
SELECT 'R2' AS metric, train_r2 AS train, val_r2 AS val FROM training_runs WHERE run_name='$run' UNION ALL SELECT 'Adj R2', train_adj_r2, val_adj_r2 FROM training_runs WHERE run_name='$run';

-- @query: hyperparams
SELECT kv.key AS parametro, kv.value AS valor FROM training_runs tr, jsonb_each_text(tr.best_params) AS kv WHERE tr.run_name='$run' ORDER BY kv.key;

-- @query: feature_importances
SELECT feature, importance FROM training_feature_importance WHERE run_name='$run' ORDER BY importance DESC;

-- @query: val_rmse_history
SELECT started_at AS "time", val_rmse FROM training_runs ORDER BY started_at;

-- @query: all_runs
SELECT run_name, started_at, round(duration_seconds::numeric,1) AS duration_s, dataset_rows, round(val_rmse::numeric,2) AS val_rmse, round(val_r2::numeric,4) AS val_r2, round(overfitting_score::numeric,4) AS overfitting, is_champion FROM training_runs ORDER BY started_at DESC;
