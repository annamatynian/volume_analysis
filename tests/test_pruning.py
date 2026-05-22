"""
tests/test_pruning.py
=====================
Unit tests for prune_old_zips() in pruning.py.

Контракт prune_old_zips(directory, keep_days, dry_run=False) -> dict:
- Ищет ZIP-файлы в directory (не рекурсивно).
- Файлы с датой YYYY-MM-DD в имени, старше (today - keep_days) — удаляются.
- Файлы без распознаваемой даты в имени — оставляются (kept), не трогаются.
- dry_run=True: возвращает результат без удаления файлов.
- dry_run=False: реально удаляет файлы.
- Возвращает: {'deleted': [...], 'kept': [...], 'errors': [...]}

Принципы тестирования:
- Тесты используют реальную файловую систему (tmp_path pytest-фикстура).
- Нет hardcoded дат — всё через datetime.today() ± delta.
- Тесты не дублируют логику (не проверяют "дата > X" внутри).
- dry_run тест проверяет через filesystem — не через возвращаемое значение.
"""

import os
import sys
import zipfile
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pruning import prune_old_zips


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_zip(directory: str, date: datetime, prefix: str = "BTCUSDT-aggTrades") -> str:
    """
    Создаёт реальный (минимальный) ZIP-файл с датой в имени.
    Имя: {prefix}-YYYY-MM-DD.zip
    """
    date_str = date.strftime('%Y-%m-%d')
    filename = f"{prefix}-{date_str}.zip"
    path = os.path.join(directory, filename)
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr("dummy.csv", "a,b\n1,2")
    return path


def _make_zip_no_date(directory: str, name: str = "BTCUSDT-unknown.zip") -> str:
    """
    Создаёт ZIP без распознаваемой даты в имени.
    Контракт: такие файлы должны остаться нетронутыми.
    """
    path = os.path.join(directory, name)
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr("dummy.csv", "a,b\n1,2")
    return path


def _days_ago(n: int) -> datetime:
    """Дата n дней назад от сегодня."""
    return datetime.today() - timedelta(days=n)


def _days_from_now(n: int) -> datetime:
    """Дата n дней в будущем (для тестов граничных случаев)."""
    return datetime.today() + timedelta(days=n)


# ---------------------------------------------------------------------------
# Тест 1: Возвращаемая структура
# ---------------------------------------------------------------------------

class TestPruneOldZipsReturnStructure:

    def test_returns_dict_with_required_keys(self, tmp_path):
        """
        Контракт: результат — dict с ключами 'deleted', 'kept', 'errors'.
        """
        result = prune_old_zips(str(tmp_path), keep_days=45)
        assert isinstance(result, dict), "Must return dict"
        assert 'deleted' in result, "Missing key 'deleted'"
        assert 'kept'    in result, "Missing key 'kept'"
        assert 'errors'  in result, "Missing key 'errors'"

    def test_all_values_are_lists(self, tmp_path):
        """
        Контракт: deleted, kept, errors — это списки (list).
        """
        result = prune_old_zips(str(tmp_path), keep_days=45)
        assert isinstance(result['deleted'], list)
        assert isinstance(result['kept'],    list)
        assert isinstance(result['errors'],  list)

    def test_empty_directory_returns_empty_lists(self, tmp_path):
        """
        Контракт: пустая директория → все три списка пустые.
        """
        result = prune_old_zips(str(tmp_path), keep_days=45)
        assert result == {'deleted': [], 'kept': [], 'errors': []}


# ---------------------------------------------------------------------------
# Тест 2: Логика удаления по дате
# ---------------------------------------------------------------------------

class TestPruneOldZipsDeletion:

    def test_old_zip_is_deleted(self, tmp_path):
        """
        Контракт: ZIP старше keep_days удаляется из FS и попадает в 'deleted'.
        WHY: накопленные ZIP занимают сотни MB → цель этапа В.
        """
        old_zip = _make_zip(str(tmp_path), _days_ago(50))  # 50 > 45
        result = prune_old_zips(str(tmp_path), keep_days=45, dry_run=False)

        assert os.path.basename(old_zip) in [os.path.basename(p) for p in result['deleted']], (
            f"Old ZIP must be in 'deleted', got {result['deleted']}"
        )
        assert not os.path.exists(old_zip), (
            "Old ZIP must be physically deleted from filesystem"
        )

    def test_recent_zip_is_kept(self, tmp_path):
        """
        Контракт: ZIP младше keep_days остаётся нетронутым и попадает в 'kept'.
        """
        recent_zip = _make_zip(str(tmp_path), _days_ago(10))  # 10 < 45
        result = prune_old_zips(str(tmp_path), keep_days=45, dry_run=False)

        assert os.path.basename(recent_zip) in [os.path.basename(p) for p in result['kept']], (
            f"Recent ZIP must be in 'kept', got {result['kept']}"
        )
        assert os.path.exists(recent_zip), (
            "Recent ZIP must NOT be deleted"
        )

    def test_boundary_exact_keep_days_is_kept(self, tmp_path):
        """
        Контракт: ZIP ровно на границе (keep_days дней назад) — ОСТАВЛЯЕТСЯ.
        WHY: удаляем только СТРОГО СТАРШЕ keep_days. Граничный файл = ещё актуален.
        Это предотвращает случайное удаление при запуске ровно в день перехода.
        """
        boundary_zip = _make_zip(str(tmp_path), _days_ago(45))  # ровно 45 дней
        result = prune_old_zips(str(tmp_path), keep_days=45, dry_run=False)

        assert os.path.exists(boundary_zip), (
            "ZIP exactly at boundary (keep_days) must NOT be deleted (strict >)"
        )
        assert os.path.basename(boundary_zip) in [os.path.basename(p) for p in result['kept']], (
            f"Boundary ZIP must be in 'kept', got {result['kept']}"
        )

    def test_mixed_old_and_recent(self, tmp_path):
        """
        Контракт: несколько ZIP разных возрастов — удаляются только старые.
        """
        old_zip1    = _make_zip(str(tmp_path), _days_ago(60), "BTCUSDT-aggTrades")
        old_zip2    = _make_zip(str(tmp_path), _days_ago(90), "BTCUSDT-1m")
        recent_zip  = _make_zip(str(tmp_path), _days_ago(5),  "BTCUSDT-aggTrades")

        result = prune_old_zips(str(tmp_path), keep_days=45, dry_run=False)

        deleted_names = [os.path.basename(p) for p in result['deleted']]
        kept_names    = [os.path.basename(p) for p in result['kept']]

        assert os.path.basename(old_zip1) in deleted_names
        assert os.path.basename(old_zip2) in deleted_names
        assert os.path.basename(recent_zip) in kept_names
        assert not os.path.exists(old_zip1)
        assert not os.path.exists(old_zip2)
        assert os.path.exists(recent_zip)

    def test_no_zip_files_only_parquet(self, tmp_path):
        """
        Контракт: parquet-файлы не трогаются (функция работает только с ZIP).
        WHY: parquet маленький, пересчёт дорогой — parquet всегда оставляем.
        """
        parquet_path = os.path.join(str(tmp_path), "BTCUSDT-anchor-2025-01.parquet")
        with open(parquet_path, 'w') as f:
            f.write("dummy")

        result = prune_old_zips(str(tmp_path), keep_days=45, dry_run=False)

        assert os.path.exists(parquet_path), "Parquet file must not be touched"
        assert result['deleted'] == []
        assert result['kept'] == []
        assert result['errors'] == []


# ---------------------------------------------------------------------------
# Тест 3: dry_run=True — ничего не удаляет
# ---------------------------------------------------------------------------

class TestPruneOldZipsDryRun:

    def test_dry_run_does_not_delete_files(self, tmp_path):
        """
        Контракт: dry_run=True — файлы физически не удаляются.
        WHY: dry_run позволяет проверить что будет удалено перед реальным запуском.
        """
        old_zip = _make_zip(str(tmp_path), _days_ago(60))
        prune_old_zips(str(tmp_path), keep_days=45, dry_run=True)

        assert os.path.exists(old_zip), (
            "dry_run=True must NOT delete any files physically"
        )

    def test_dry_run_reports_would_delete(self, tmp_path):
        """
        Контракт: dry_run=True возвращает корректный список 'deleted'
        (что БЫЛО БЫ удалено).
        """
        old_zip = _make_zip(str(tmp_path), _days_ago(60))
        result  = prune_old_zips(str(tmp_path), keep_days=45, dry_run=True)

        assert os.path.basename(old_zip) in [os.path.basename(p) for p in result['deleted']], (
            f"dry_run must report old ZIP as would-be-deleted, got {result['deleted']}"
        )

    def test_dry_run_reports_would_keep(self, tmp_path):
        """
        Контракт: dry_run=True корректно репортит 'kept'.
        """
        recent_zip = _make_zip(str(tmp_path), _days_ago(10))
        result     = prune_old_zips(str(tmp_path), keep_days=45, dry_run=True)

        assert os.path.basename(recent_zip) in [os.path.basename(p) for p in result['kept']], (
            f"dry_run must report recent ZIP as would-be-kept, got {result['kept']}"
        )

    def test_dry_run_false_actually_deletes(self, tmp_path):
        """
        Контракт: dry_run=False (дефолт) — реально удаляет.
        Убеждаемся что dry_run=False != dry_run=True по эффекту на FS.
        """
        old_zip = _make_zip(str(tmp_path), _days_ago(60))
        prune_old_zips(str(tmp_path), keep_days=45, dry_run=False)

        assert not os.path.exists(old_zip), (
            "dry_run=False must actually delete old files"
        )


# ---------------------------------------------------------------------------
# Тест 4: файлы без даты в имени
# ---------------------------------------------------------------------------

class TestPruneOldZipsNoDate:

    def test_zip_without_date_is_kept(self, tmp_path):
        """
        Контракт: ZIP без даты YYYY-MM-DD в имени — оставляется нетронутым.
        WHY: не можем безопасно определить возраст → лучше не трогать.
        """
        no_date_zip = _make_zip_no_date(str(tmp_path), "BTCUSDT-unknown.zip")
        result = prune_old_zips(str(tmp_path), keep_days=45, dry_run=False)

        assert os.path.exists(no_date_zip), (
            "ZIP without date in name must not be deleted"
        )
        assert os.path.basename(no_date_zip) in [os.path.basename(p) for p in result['kept']], (
            f"ZIP without date must be in 'kept', got {result['kept']}"
        )

    def test_non_zip_files_ignored_entirely(self, tmp_path):
        """
        Контракт: не-ZIP файлы (txt, csv, parquet) игнорируются полностью —
        не попадают ни в deleted, ни в kept.
        WHY: функция прунит только ZIP, остальное вне её ответственности.
        """
        txt_path = os.path.join(str(tmp_path), "notes.txt")
        with open(txt_path, 'w') as f:
            f.write("ignore me")

        result = prune_old_zips(str(tmp_path), keep_days=45, dry_run=False)

        all_reported = result['deleted'] + result['kept'] + result['errors']
        assert not any('notes.txt' in p for p in all_reported), (
            "Non-ZIP files must be completely ignored"
        )
        assert os.path.exists(txt_path), "Non-ZIP files must not be deleted"


# ---------------------------------------------------------------------------
# Тест 5: кастомный keep_days
# ---------------------------------------------------------------------------

class TestPruneOldZipsKeepDays:

    def test_keep_days_7_deletes_8_day_old(self, tmp_path):
        """
        Контракт: keep_days=7 → файл 8-дневной давности удаляется.
        """
        old_zip = _make_zip(str(tmp_path), _days_ago(8))
        result = prune_old_zips(str(tmp_path), keep_days=7, dry_run=False)

        assert os.path.basename(old_zip) in [os.path.basename(p) for p in result['deleted']]
        assert not os.path.exists(old_zip)

    def test_keep_days_0_deletes_all_dated_zips(self, tmp_path):
        """
        Контракт: keep_days=0 → все файлы с датой ≥ 1 день назад удаляются.
        WHY: граничный случай — keep_days=0 означает "хранить только сегодня".
        """
        old_zip = _make_zip(str(tmp_path), _days_ago(1))
        result = prune_old_zips(str(tmp_path), keep_days=0, dry_run=False)

        assert os.path.basename(old_zip) in [os.path.basename(p) for p in result['deleted']]
        assert not os.path.exists(old_zip)

    def test_keep_days_default_is_45(self, tmp_path):
        """
        Контракт: вызов без keep_days эквивалентен keep_days=45.
        """
        # 46-дневный файл должен удалиться с дефолтным keep_days
        old_zip = _make_zip(str(tmp_path), _days_ago(46))
        result = prune_old_zips(str(tmp_path), dry_run=True)  # dry_run — не трогаем FS

        assert os.path.basename(old_zip) in [os.path.basename(p) for p in result['deleted']], (
            f"Default keep_days=45 must delete 46-day-old ZIP, got kept={result['kept']}"
        )
