"""Valida que las queries SQL de los dashboards de Grafana ejecuten sin error.

Lee los .sql de sql_test/queries/, sustituye los placeholders `$run` y `$batch`
por valores reales tomados de la DB, ejecuta cada query contra Postgres y reporta
el numero de filas devueltas. Sale con codigo != 0 si alguna query falla.

Uso:
    docker compose up -d postgres        # asegurar Postgres arriba
    python sql_test/run_tests.py         # usa DB_* (default localhost:5432)

Variables de entorno (con defaults): DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME.
"""
import os
import re
import sys
import glob

from sqlalchemy import create_engine, text


def get_engine():
    cfg = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'user': os.getenv('DB_USER', 'metlife_user'),
        'password': os.getenv('DB_PASSWORD', 'metlife_pass'),
        'database': os.getenv('DB_NAME', 'metlife_db'),
    }
    url = (f"postgresql://{cfg['user']}:{cfg['password']}"
           f"@{cfg['host']}:{cfg['port']}/{cfg['database']}")
    return create_engine(url)


def parse_queries(path):
    """Devuelve lista de (nombre, sql) a partir de un archivo con marcadores
    `-- @query: <nombre>`.
    """
    with open(path, encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'(?m)^--\s*@query:\s*(.+)$', content)
    # blocks = [preamble, name1, body1, name2, body2, ...]
    queries = []
    for i in range(1, len(blocks), 2):
        name = blocks[i].strip()
        body = blocks[i + 1]
        # quitar comentarios de linea y normalizar
        sql_lines = [ln for ln in body.splitlines() if not ln.strip().startswith('--')]
        sql = '\n'.join(sql_lines).strip().rstrip(';').strip()
        if sql:
            queries.append((name, sql))
    return queries


def fetch_sample(engine, sql):
    try:
        with engine.connect() as conn:
            row = conn.execute(text(sql)).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def main():
    engine = get_engine()

    sample_run = fetch_sample(
        engine, "SELECT run_name FROM training_runs ORDER BY started_at DESC LIMIT 1")
    sample_batch = fetch_sample(
        engine, "SELECT batch FROM prod_predictions ORDER BY batch LIMIT 1")

    print(f"Sample run_name: {sample_run!r}")
    print(f"Sample batch:    {sample_batch!r}")
    if sample_run is None:
        print("  (aviso) training_runs vacio: las queries con $run devolveran 0 filas.")
    if sample_batch is None:
        print("  (aviso) prod_predictions vacio: las queries con $batch devolveran 0 filas.")
    print("-" * 70)

    run_val = sample_run if sample_run is not None else ''
    batch_val = sample_batch if sample_batch is not None else ''

    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(base_dir, 'queries', '*.sql')))

    total = 0
    failures = 0
    for path in files:
        print(f"\n=== {os.path.basename(path)} ===")
        for name, sql in parse_queries(path):
            total += 1
            final_sql = sql.replace('$run', run_val).replace('$batch', batch_val)
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(final_sql))
                    n = len(result.fetchall())
                print(f"  [OK]   {name:24s} -> {n} filas")
            except Exception as e:
                failures += 1
                print(f"  [FAIL] {name:24s} -> {e}")

    print("\n" + "=" * 70)
    print(f"Total queries: {total} | OK: {total - failures} | FAIL: {failures}")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
