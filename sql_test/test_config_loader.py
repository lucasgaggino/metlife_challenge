"""Validacion rapida de config.yaml."""
import sys
sys.path.insert(0, 'src')

from config_loader import load_config, resolve_promotion_metric_value, get_prod_batches
from monitoring import apply_monitoring_config, PSI_WARNING

load_config(force_reload=True)
apply_monitoring_config()

cfg = load_config()
assert cfg['model']['type'] == 'xgboost'
assert PSI_WARNING == 0.10
assert len(get_prod_batches()) == 3

m = resolve_promotion_metric_value({
    'validation': {'rmse': 100.0, 'r2': 0.9},
})
assert m == 100.0

print('config_loader OK')
