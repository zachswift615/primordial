import pytest
import tempfile
import os
from primordial.experiments.metrics_collector import MetricsCollector


def test_metrics_collector_creation():
    collector = MetricsCollector()
    assert collector.metrics == []


def test_metrics_collector_record():
    collector = MetricsCollector()

    collector.record({'step': 1, 'survival_time': 10.0})
    collector.record({'step': 2, 'survival_time': 20.0})

    assert len(collector.metrics) == 2


def test_metrics_collector_summary():
    collector = MetricsCollector()

    for i in range(10):
        collector.record({'step': i, 'survival_time': float(i * 10)})

    summary = collector.summary()

    assert 'mean_survival_time' in summary
    assert 'max_survival_time' in summary
    assert summary['max_survival_time'] == 90.0


def test_metrics_collector_export_csv():
    collector = MetricsCollector()

    collector.record({'step': 1, 'survival_time': 10.0})
    collector.record({'step': 2, 'survival_time': 20.0})

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        path = f.name

    try:
        collector.export_csv(path)

        with open(path, 'r') as f:
            content = f.read()

        assert 'step' in content
        assert 'survival_time' in content
    finally:
        os.unlink(path)


def test_metrics_collector_export_json():
    collector = MetricsCollector()

    collector.record({'step': 1, 'value': 10.0})

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name

    try:
        collector.export_json(path)

        import json
        with open(path, 'r') as f:
            data = json.load(f)

        assert len(data['metrics']) == 1
    finally:
        os.unlink(path)
