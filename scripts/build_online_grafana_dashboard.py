"""Genera grafana/dashboards/online_predictions.json desde prod_predictions.json."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'grafana' / 'dashboards' / 'prod_predictions.json'
DST = ROOT / 'grafana' / 'dashboards' / 'online_predictions.json'

REMOVE_TITLES = {
    'Predicho vs Real (solo si hay target)',
    'Residuales (real - predicho)',
}


def transform_sql(sql: str) -> str:
    if not sql:
        return sql
    sql = sql.replace('prod_predictions', 'online_predictions')
    sql = sql.replace("batch='$batch'", "session_id='$session'")
    sql = sql.replace("batch = '$batch'", "session_id = '$session'")
    sql = re.sub(
        r"FROM baseline_predictions\b",
        "FROM baseline_predictions WHERE environment = '$environment'",
        sql,
        flags=re.IGNORECASE,
    )
    if 'online_predictions' in sql and "environment = '$environment'" not in sql:
        sql = re.sub(
            r'(FROM\s+online_predictions)(\s+)',
            r"\1 WHERE environment = '$environment'\2",
            sql,
            count=1,
            flags=re.IGNORECASE,
        )
        sql = re.sub(
            r'(FROM\s+online_predictions\s+WHERE\s+environment\s*=\s*\'\$environment\'\s+WHERE)',
            r'FROM online_predictions WHERE environment = \'$environment\' AND',
            sql,
            flags=re.IGNORECASE,
        )
    if 'monitoring_runs' in sql and "environment = '$environment'" not in sql:
        sql = re.sub(
            r'(FROM\s+monitoring_runs)(\s+)',
            r"\1 WHERE environment = '$environment'\2",
            sql,
            count=1,
            flags=re.IGNORECASE,
        )
    return sql


def patch_obj(obj):
    if isinstance(obj, dict):
        if obj.get('title') in REMOVE_TITLES:
            return None
        if obj.get('title') == 'Predicciones: batch vs baseline':
            obj['title'] = 'Predicciones: online vs baseline (entrenamiento)'
        if obj.get('title') == 'Estado de monitoreo: $batch':
            obj['title'] = 'Estado de monitoreo: $session'
        for k, v in list(obj.items()):
            if k == 'rawSql' and isinstance(v, str):
                obj[k] = transform_sql(v)
            elif k in ('definition', 'query') and isinstance(v, str):
                if 'prod_predictions' in v or 'batch' in v:
                    obj[k] = transform_sql(v) if 'prod_predictions' in v else v.replace(
                        'batch', 'session_id'
                    )
            else:
                patched = patch_obj(v)
                if patched is None and k == 'panels':
                    continue
                obj[k] = patched
        if 'panels' in obj and isinstance(obj['panels'], list):
            obj['panels'] = [p for p in obj['panels'] if p is not None]
    elif isinstance(obj, list):
        out = []
        for item in obj:
            patched = patch_obj(item)
            if patched is not None:
                out.append(patched)
        return out
    return obj


def main():
    data = json.loads(SRC.read_text(encoding='utf-8'))
    data['title'] = 'Online Predictions'
    data['uid'] = 'metlife-online-predictions'
    data['tags'] = ['metlife', 'online', 'scoring']
    templating = data.get('templating', {}).get('list', [])
    for var in templating:
        if var.get('name') == 'batch':
            var['name'] = 'session'
            var['label'] = 'Session'
            q = (
                "SELECT DISTINCT session_id FROM online_predictions "
                "WHERE environment = '$environment' ORDER BY session_id DESC"
            )
            var['definition'] = q
            var['query'] = q
    # Simplify monitoring stat SQL (no RMSE when no target)
    for panel in data.get('panels', []):
        if panel.get('title', '').startswith('Estado de monitoreo'):
            for t in panel.get('targets', []):
                t['rawSql'] = (
                    "SELECT status AS \"Estado\", n_rows AS \"Filas\", "
                    "round(max_psi::numeric,4) AS \"Max PSI\", "
                    "round(prediction_psi::numeric,4) AS \"Pred PSI\", "
                    "round(schema_violation_pct::numeric,2) AS \"Schema viol %\", "
                    "drift_status AS \"Drift\", prediction_drift_status AS \"Pred drift\", "
                    "schema_status AS \"Schema\" "
                    "FROM monitoring_runs WHERE environment = '$environment' "
                    "AND batch='$session' ORDER BY run_time DESC LIMIT 1"
                )
    data = patch_obj(data)
    DST.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {DST}')


if __name__ == '__main__':
    main()
