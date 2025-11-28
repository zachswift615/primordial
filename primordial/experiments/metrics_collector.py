"""Metrics collection and export for experiments."""

import json
import csv
from typing import Dict, Any, List
from pathlib import Path


class MetricsCollector:
    """Collects and exports experiment metrics.

    Provides utilities for:
    - Recording metrics over time
    - Computing summary statistics
    - Exporting to CSV and JSON
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: List[Dict[str, Any]] = []

    def record(self, metrics: Dict[str, Any]) -> None:
        """Record a metrics snapshot.

        Args:
            metrics: Dictionary of metric values.
        """
        self.metrics.append(metrics.copy())

    def clear(self) -> None:
        """Clear all recorded metrics."""
        self.metrics = []

    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics.

        Returns:
            Dictionary with mean, max, min for numeric fields.
        """
        if not self.metrics:
            return {}

        summary = {}

        # Find all numeric fields
        numeric_fields = []
        for key, value in self.metrics[0].items():
            if isinstance(value, (int, float)):
                numeric_fields.append(key)

        # Compute statistics
        for field in numeric_fields:
            values = [m[field] for m in self.metrics if field in m]
            if values:
                summary[f'mean_{field}'] = sum(values) / len(values)
                summary[f'max_{field}'] = max(values)
                summary[f'min_{field}'] = min(values)

        summary['total_records'] = len(self.metrics)

        return summary

    def export_csv(self, path: str) -> None:
        """Export metrics to CSV file.

        Args:
            path: Output file path.
        """
        if not self.metrics:
            return

        # Collect all fieldnames
        fieldnames = set()
        for m in self.metrics:
            fieldnames.update(m.keys())
        fieldnames = sorted(fieldnames)

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics)

    def export_json(self, path: str) -> None:
        """Export metrics to JSON file.

        Args:
            path: Output file path.
        """
        data = {
            'metrics': self.metrics,
            'summary': self.summary()
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_field_over_time(self, field: str) -> List[Any]:
        """Get values of a field over time.

        Args:
            field: Field name.

        Returns:
            List of values.
        """
        return [m.get(field) for m in self.metrics]
