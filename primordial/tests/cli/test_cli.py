import pytest
import subprocess
import sys


def test_cli_help():
    """Test that CLI help works."""
    result = subprocess.run(
        [sys.executable, '-m', 'primordial', '--help'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert 'Primordial' in result.stdout


def test_cli_experiment_list():
    """Test listing experiments."""
    result = subprocess.run(
        [sys.executable, '-m', 'primordial', 'experiment', '--list'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert 'survival' in result.stdout.lower()
