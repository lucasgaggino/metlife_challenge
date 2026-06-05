"""Shared fixtures for unit tests."""
import pandas as pd
import pytest


@pytest.fixture
def sample_features_df():
    return pd.DataFrame({
        'age': [25, 45, 60],
        'sex': ['male', 'female', 'male'],
        'bmi': [22.0, 30.5, 27.0],
        'children': [0, 2, 1],
        'smoker': ['no', 'yes', 'no'],
        'region': ['southwest', 'northwest', 'southeast'],
    })


@pytest.fixture
def sample_training_df(sample_features_df):
    df = sample_features_df.copy()
    df['charges'] = [5000.0, 25000.0, 12000.0]
    df['id'] = [1, 2, 3]
    df['created_at'] = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
    return df


@pytest.fixture
def schema_test_config(monkeypatch):
    """Override monitoring schema globals for isolated check_schema tests."""
    import monitoring

    monkeypatch.setattr(monitoring, 'NUMERIC_RANGES', {
        'age': (18.0, 100.0),
        'bmi': (10.0, 60.0),
    })
    monkeypatch.setattr(monitoring, 'CATEGORICAL_DOMAINS', {
        'sex': {'male', 'female'},
        'smoker': {'yes', 'no'},
    })
    monkeypatch.setattr(monitoring, 'SCHEMA_WARNING_PCT', 0.0)
    monkeypatch.setattr(monitoring, 'SCHEMA_ALERT_PCT', 5.0)
