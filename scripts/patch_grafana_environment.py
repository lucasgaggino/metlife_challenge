"""Añade variable $environment y filtros SQL en dashboards Grafana."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / 'grafana' / 'dashboards'

ENV_VAR = {
    'current': {'selected': True, 'text': 'sandbox', 'value': 'sandbox'},
    'hide': 0,
    'includeAll': False,
    'label': 'Environment',
    'multi': False,
    'name': 'environment',
    'options': [
        {'selected': True, 'text': 'sandbox', 'value': 'sandbox'},
        {'selected': False, 'text': 'prod', 'value': 'prod'},
    ],
    'query': 'sandbox,prod',
    'type': 'custom',
}

TABLES = (
    'prod_predictions',
    'monitoring_runs',
    'baseline_predictions',
    'training_runs',
    'training_feature_importance',
)


def _inject_env_filter(sql: str) -> str:
    if not sql or '$environment' in sql:
        return sql
    for table in TABLES:
        if table not in sql.lower():
            continue
        pattern = re.compile(
            rf'(\bFROM\s+{table}\b)(\s+)(?!WHERE\b)',
            re.IGNORECASE,
        )
        sql, n = pattern.subn(
            rf"\1 WHERE environment = '$environment'\2",
            sql,
            count=1,
        )
        if n:
            continue
        pattern2 = re.compile(
            rf'(\bFROM\s+{table}\s+WHERE\s+)',
            re.IGNORECASE,
        )
        sql, n2 = pattern2.subn(
            rf"\1environment = '$environment' AND ",
            sql,
            count=1,
        )
    return sql


def _patch_queries(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == 'rawSql' and isinstance(v, str):
                obj[k] = _inject_env_filter(v)
            elif k in ('definition', 'query') and isinstance(v, str) and any(
                t in v.lower() for t in TABLES
            ):
                obj[k] = _inject_env_filter(v)
            else:
                _patch_queries(v)
    elif isinstance(obj, list):
        for item in obj:
            _patch_queries(item)


def patch_file(path: Path) -> None:
    data = json.loads(path.read_text(encoding='utf-8'))
    templating = data.setdefault('templating', {}).setdefault('list', [])
    if not any(v.get('name') == 'environment' for v in templating):
        templating.insert(0, ENV_VAR)
    _patch_queries(data)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Patched {path.name}')


def main():
    for p in sorted(ROOT.glob('*.json')):
        patch_file(p)


if __name__ == '__main__':
    main()
