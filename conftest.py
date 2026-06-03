"""
conftest.py
===========
Project-wide pytest configuration.
Ensures all temp/cache files go to D: not C:

# WHY collect_ignore: test_volume_density_backup_before_etap8.py зависает при полном pytest
# (58с без ни одного теста). Сохраняется для истории; от сборки отключён через conftest.
collect_ignore = ["tests/test_volume_density_backup_before_etap8.py"]
"""

import os
import tempfile
import warnings
import pytest

# WHY collect_ignore: test_macro_client.py требует fredapi (не установлен в venv);
# ошибка импорта прерывает весь pytest сьюрх.
# NB: чтобы запустить эти тесты: pip install fredapi + $env:FRED_API_KEY=...
collect_ignore = [
    "tests/test_volume_density_backup_before_etap8.py",
    "tests/test_macro_client.py",
    "tests/test_mozart_llm_smoke.py",
]

# Override Windows %TEMP% for this pytest session.
# Prevents sklearn, joblib, and other libs from writing to C:/Users/.../AppData/Local/Temp
_PROJECT_TMP = "D:/DeFi-RAG-Projects/volume_analysis/.tmp"
os.makedirs(_PROJECT_TMP, exist_ok=True)
os.environ["TEMP"]   = _PROJECT_TMP
os.environ["TMP"]    = _PROJECT_TMP
os.environ["TMPDIR"] = _PROJECT_TMP
tempfile.tempdir     = _PROJECT_TMP

# Suppress DeprecationWarning from pytz (third-party lib, not our code).
# pytz uses datetime.utcfromtimestamp() internally which is deprecated in Python 3.12+.
# Remove this filter when pytz drops support for the old API or is replaced by zoneinfo.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"pytz\.tzinfo",
)


@pytest.fixture(scope="session")
def tmp_path_factory_base(tmp_path_factory):
    """All tmp_path fixtures also land on D: drive."""
    return tmp_path_factory.mktemp("session")
