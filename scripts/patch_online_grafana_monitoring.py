"""Reemplaza panel de monitoreo estático por gauges + time series en online_predictions.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / 'grafana' / 'dashboards' / 'online_predictions.json'

ENV_SESSION = "environment = '$environment' AND session_id = '$session'"
SNAP = 'online_monitoring_snapshots'
PSI_TBL = 'online_monitoring_psi'
ALERTS = 'online_retrain_alerts'


def _pg_ds():
    return {'type': 'postgres', 'uid': 'metlife_pg'}


def _sql_target(fmt: str, sql: str):
    return {
        'datasource': _pg_ds(),
        'editorMode': 'code',
        'format': fmt,
        'rawQuery': True,
        'rawSql': sql,
        'refId': 'A',
    }


def stat_panel(pid, title, sql, x, y, w=4, h=4):
    return {
        'datasource': _pg_ds(),
        'fieldConfig': {
            'defaults': {
                'color': {'mode': 'thresholds'},
                'mappings': [],
                'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}]},
            },
            'overrides': [],
        },
        'gridPos': {'h': h, 'w': w, 'x': x, 'y': y},
        'id': pid,
        'options': {
            'colorMode': 'value',
            'graphMode': 'none',
            'justifyMode': 'auto',
            'orientation': 'auto',
            'reduceOptions': {'calcs': ['lastNotNull'], 'fields': '', 'values': False},
            'textMode': 'auto',
        },
        'pluginVersion': '11.3.0',
        'targets': [_sql_target('table', sql)],
        'title': title,
        'type': 'stat',
    }


def gauge_panel(pid, title, sql, x, y, w=5, h=4):
    return {
        'datasource': _pg_ds(),
        'fieldConfig': {
            'defaults': {
                'color': {'mode': 'thresholds'},
                'mappings': [],
                'max': 0.5,
                'min': 0,
                'thresholds': {
                    'mode': 'absolute',
                    'steps': [
                        {'color': 'green', 'value': None},
                        {'color': 'yellow', 'value': 0.1},
                        {'color': 'red', 'value': 0.25},
                    ],
                },
                'unit': 'none',
            },
            'overrides': [],
        },
        'gridPos': {'h': h, 'w': w, 'x': x, 'y': y},
        'id': pid,
        'options': {
            'minVizHeight': 75,
            'minVizWidth': 75,
            'orientation': 'auto',
            'reduceOptions': {'calcs': ['lastNotNull'], 'fields': '', 'values': False},
            'showThresholdLabels': False,
            'showThresholdMarkers': True,
            'sizing': 'auto',
        },
        'pluginVersion': '11.3.0',
        'targets': [_sql_target('table', sql)],
        'title': title,
        'type': 'gauge',
    }


def timeseries_panel(pid, title, sql, x, y, w=12, h=7, extra_targets=None):
    targets = [_sql_target('time_series', sql)]
    if extra_targets:
        targets.extend(extra_targets)
    return {
        'datasource': _pg_ds(),
        'fieldConfig': {
            'defaults': {
                'color': {'mode': 'palette-classic'},
                'custom': {
                    'axisBorderShow': False,
                    'axisCenteredZero': False,
                    'axisColorMode': 'text',
                    'axisPlacement': 'auto',
                    'drawStyle': 'line',
                    'fillOpacity': 10,
                    'lineWidth': 2,
                    'showPoints': 'auto',
                    'spanNulls': False,
                },
                'mappings': [],
                'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}]},
            },
            'overrides': [
                {
                    'matcher': {'id': 'byRegexp', 'options': 'retrain|alert|psi_alert|psi_warn'},
                    'properties': [
                        {
                            'id': 'custom.drawStyle',
                            'value': 'points',
                        },
                        {
                            'id': 'custom.pointSize',
                            'value': 10,
                        },
                        {
                            'id': 'custom.lineWidth',
                            'value': 0,
                        },
                    ],
                },
            ],
        },
        'gridPos': {'h': h, 'w': w, 'x': x, 'y': y},
        'id': pid,
        'options': {
            'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom', 'showLegend': True},
            'tooltip': {'mode': 'multi', 'sort': 'none'},
        },
        'pluginVersion': '11.3.0',
        'targets': targets,
        'title': title,
        'type': 'timeseries',
    }


def main():
    data = json.loads(DASH.read_text(encoding='utf-8'))
    panels = data.get('panels', [])

    # Quitar panel antiguo "Estado de monitoreo"
    panels = [p for p in panels if not (
        p.get('title', '').startswith('Estado de monitoreo')
    )]

    # Desplazar paneles existentes hacia abajo (+11 en y)
    for p in panels:
        gp = p.get('gridPos', {})
        if gp.get('y', 0) >= 5:
            gp['y'] = gp.get('y', 0) + 11

    filas_sql = f"""
        SELECT COALESCE(
            (SELECT MAX(end_request_seq)+1 FROM {SNAP}
             WHERE {ENV_SESSION}),
            (SELECT COUNT(*)::int FROM online_predictions
             WHERE {ENV_SESSION})
        ) AS filas
    """.strip()

    pred_gauge_sql = f"""
        SELECT prediction_psi FROM {SNAP}
        WHERE {ENV_SESSION}
        ORDER BY measured_at DESC LIMIT 1
    """.strip()

    max_gauge_sql = f"""
        SELECT max_psi FROM {SNAP}
        WHERE {ENV_SESSION}
        ORDER BY measured_at DESC LIMIT 1
    """.strip()

    pred_ts_sql = f"""
        SELECT measured_at AS time, prediction_psi AS value
        FROM {SNAP}
        WHERE {ENV_SESSION} AND prediction_psi IS NOT NULL
        ORDER BY measured_at
    """.strip()

    retrain_alert_ts_sql = f"""
        SELECT measured_at AS time, prediction_psi AS value, alert_type AS metric
        FROM {ALERTS}
        WHERE {ENV_SESSION}
        ORDER BY measured_at
    """.strip()

    psi_ts_sql = f"""
        SELECT measured_at AS time, psi AS value, feature AS metric
        FROM {PSI_TBL}
        WHERE {ENV_SESSION}
        ORDER BY measured_at
    """.strip()

    schema_ts_sql = f"""
        SELECT measured_at AS time, schema_violation_pct AS value
        FROM {SNAP}
        WHERE {ENV_SESSION}
        ORDER BY measured_at
    """.strip()

    new_panels = [
        stat_panel(100, 'Filas analizadas (run)', filas_sql, 0, 0, w=4),
        gauge_panel(101, 'Prediction PSI (actual)', pred_gauge_sql, 4, 0),
        gauge_panel(102, 'Max PSI covariables (actual)', max_gauge_sql, 9, 0),
        timeseries_panel(
            103,
            'Prediction PSI en el tiempo (alertas retrain)',
            pred_ts_sql,
            0,
            4,
            extra_targets=[_sql_target('time_series', retrain_alert_ts_sql)],
        ),
        timeseries_panel(104, 'Evolucion PSI por feature', psi_ts_sql, 12, 4),
        timeseries_panel(105, 'Schema violation %', schema_ts_sql, 0, 11, w=24, h=5),
    ]

    data['panels'] = new_panels + panels
    DASH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Updated {DASH}')


if __name__ == '__main__':
    main()
