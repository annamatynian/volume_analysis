"""
tests/test_download_anchor_data.py
===================================
Unit tests for download_anchor_data.py — Этап 8D Шаг 2.

Тестируем только чистые функции без реальных сетевых вызовов:
  - build_anchor_zip_path()  — детерминированное построение пути к ZIP
  - download_anchor_month()  — скачивание одного месяца (мок urllib)

Принципы:
- Сетевые вызовы мокируются через unittest.mock — тест не зависит от интернета.
- Тесты проверяют контракт (что возвращается / какие файлы создаются), не реализацию.
- Все файлы создаются в tmp_path (pytest fixture) — ничего не пишется в реальный проект.
"""

import os
import zipfile
import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from download_anchor_data import build_anchor_zip_path, download_anchor_month


# ---------------------------------------------------------------------------
# build_anchor_zip_path — чистая функция, сетевых вызовов нет
# ---------------------------------------------------------------------------

class TestBuildAnchorZipPath:
    """
    Контракт build_anchor_zip_path(year, month, base_dir) -> str:
    - Возвращает абсолютный путь к ZIP-файлу.
    - Имя файла: BTCUSDT-aggTrades-{YYYY}-{MM:02d}.zip
    - Путь вложен в base_dir (не зависит от cwd).
    - Не создаёт файл/папку — только строит путь.
    """

    def test_returns_string(self, tmp_path):
        result = build_anchor_zip_path(2025, 1, str(tmp_path))
        assert isinstance(result, str)

    def test_filename_format(self, tmp_path):
        """
        Контракт: имя файла BTCUSDT-aggTrades-YYYY-MM.zip с ведущим нулём месяца.
        WHY: Binance Vision URL требует точно этот формат.
        """
        path = build_anchor_zip_path(2025, 1, str(tmp_path))
        assert os.path.basename(path) == "BTCUSDT-aggTrades-2025-01.zip"

    def test_month_zero_padded(self, tmp_path):
        """
        Контракт: однозначный месяц дополняется нулём (01..09).
        """
        path = build_anchor_zip_path(2024, 9, str(tmp_path))
        assert "2024-09" in os.path.basename(path)

    def test_path_inside_base_dir(self, tmp_path):
        """
        Контракт: путь начинается с base_dir — файл всегда в нужной папке.
        """
        path = build_anchor_zip_path(2025, 1, str(tmp_path))
        assert path.startswith(str(tmp_path))

    def test_does_not_create_file(self, tmp_path):
        """
        Контракт: функция только строит путь, не создаёт файл/папку.
        WHY: чистая функция без сайд-эффектов — создание папки задача download_.
        """
        build_anchor_zip_path(2025, 1, str(tmp_path))
        zip_path = os.path.join(str(tmp_path), "BTCUSDT-aggTrades-2025-01.zip")
        assert not os.path.exists(zip_path)

    def test_different_years_different_paths(self, tmp_path):
        """
        Контракт: разные (year, month) → разные пути.
        """
        p1 = build_anchor_zip_path(2024, 12, str(tmp_path))
        p2 = build_anchor_zip_path(2025, 1,  str(tmp_path))
        assert p1 != p2


# ---------------------------------------------------------------------------
# download_anchor_month — мокируем urllib.request.urlretrieve
# ---------------------------------------------------------------------------

class TestDownloadAnchorMonth:
    """
    Контракт download_anchor_month(year, month, out_dir, base_url) -> str | None:
    - Скачивает ZIP если файл не существует → возвращает абсолютный путь.
    - Если файл уже существует → пропускает скачивание, возвращает путь (idempotent).
    - При ошибке сети → возвращает None (не поднимает исключение).
    - Создаёт out_dir если не существует.
    - Не скачивает файл дважды (urlretrieve вызывается ровно один раз при первом запуске).
    """

    BASE_URL = "https://data.binance.vision/data/futures/um/monthly/aggTrades/BTCUSDT"

    def _make_fake_zip(self, path: str) -> None:
        """Создаёт минимальный валидный ZIP для симуляции существующего файла."""
        with zipfile.ZipFile(path, 'w') as zf:
            zf.writestr("dummy.csv", "col1,col2\n1,2\n")

    def test_returns_path_on_success(self, tmp_path):
        """
        Контракт: успешное скачивание → возвращает строку-путь к файлу.
        """
        def fake_urlretrieve(url, dest, reporthook=None):
            # Симулируем скачивание: просто создаём файл
            self._make_fake_zip(dest)

        with patch("download_anchor_data.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            result = download_anchor_month(2025, 1, str(tmp_path), self.BASE_URL)

        assert isinstance(result, str)
        assert result.endswith(".zip")

    def test_file_created_on_disk(self, tmp_path):
        """
        Контракт: после успешного скачивания файл существует на диске.
        """
        def fake_urlretrieve(url, dest, reporthook=None):
            self._make_fake_zip(dest)

        with patch("download_anchor_data.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            result = download_anchor_month(2025, 1, str(tmp_path), self.BASE_URL)

        assert os.path.exists(result)

    def test_skip_if_file_exists(self, tmp_path):
        """
        Контракт: файл уже существует → urlretrieve НЕ вызывается, путь возвращается.
        WHY: idempotent — повторный запуск не скачивает заново.
        """
        # Создаём файл заранее
        zip_path = os.path.join(str(tmp_path), "BTCUSDT-aggTrades-2025-01.zip")
        self._make_fake_zip(zip_path)

        with patch("download_anchor_data.urllib.request.urlretrieve") as mock_dl:
            result = download_anchor_month(2025, 1, str(tmp_path), self.BASE_URL)

        mock_dl.assert_not_called()
        assert result == zip_path

    def test_returns_none_on_network_error(self, tmp_path):
        """
        Контракт: ошибка сети → возвращает None (не поднимает исключение).
        WHY: оркестратор должен продолжить работу с остальными месяцами.
        """
        with patch(
            "download_anchor_data.urllib.request.urlretrieve",
            side_effect=Exception("Network error"),
        ):
            result = download_anchor_month(2025, 1, str(tmp_path), self.BASE_URL)

        assert result is None

    def test_incomplete_file_removed_on_error(self, tmp_path):
        """
        Контракт: при ошибке сети неполный файл удаляется с диска.
        WHY: неполный ZIP сломает load_aggtrades_zip() при следующем запуске.
        """
        zip_path = os.path.join(str(tmp_path), "BTCUSDT-aggTrades-2025-01.zip")

        def fail_after_partial(url, dest, reporthook=None):
            # Симулируем частичный файл перед ошибкой
            with open(dest, 'w') as f:
                f.write("partial")
            raise Exception("Connection lost")

        with patch("download_anchor_data.urllib.request.urlretrieve", side_effect=fail_after_partial):
            download_anchor_month(2025, 1, str(tmp_path), self.BASE_URL)

        assert not os.path.exists(zip_path), (
            "Partial file must be removed after download error"
        )

    def test_creates_out_dir_if_not_exists(self, tmp_path):
        """
        Контракт: out_dir создаётся автоматически если не существует.
        WHY: при первом запуске папка data/delta_cache/ может отсутствовать.
        """
        new_dir = os.path.join(str(tmp_path), "deep", "nested", "dir")
        assert not os.path.exists(new_dir)

        def fake_urlretrieve(url, dest, reporthook=None):
            self._make_fake_zip(dest)

        with patch("download_anchor_data.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            download_anchor_month(2025, 1, new_dir, self.BASE_URL)

        assert os.path.isdir(new_dir)

    def test_url_constructed_correctly(self, tmp_path):
        """
        Контракт: URL передаётся в urlretrieve в формате {base_url}/{filename}.
        WHY: Binance Vision требует точный URL — проверяем что не опечатались.
        """
        captured_url = []

        def capture_url(url, dest, reporthook=None):
            captured_url.append(url)
            self._make_fake_zip(dest)

        with patch("download_anchor_data.urllib.request.urlretrieve", side_effect=capture_url):
            download_anchor_month(2025, 1, str(tmp_path), self.BASE_URL)

        assert len(captured_url) == 1
        assert captured_url[0] == f"{self.BASE_URL}/BTCUSDT-aggTrades-2025-01.zip"


# ---------------------------------------------------------------------------
# download_reaction_month — Шаг 1 Reaction Period
# ---------------------------------------------------------------------------

from download_anchor_data import download_reaction_month, build_reaction_zip_path


class TestBuildReactionZipPath:
    """
    Контракт build_reaction_zip_path(year, month, day, base_dir) -> str:
    - Возвращает путь к daily ZIP-файлу aggTrades.
    - Имя файла: BTCUSDT-aggTrades-YYYY-MM-DD.zip
    - Путь вложен в base_dir.
    - Не создаёт файл/папку — только строит путь.
    """

    def test_returns_string(self, tmp_path):
        result = build_reaction_zip_path(2025, 4, 1, str(tmp_path))
        assert isinstance(result, str)

    def test_filename_format(self, tmp_path):
        """
        Контракт: имя файла BTCUSDT-aggTrades-YYYY-MM-DD.zip с ведущими нулями.
        WHY: Binance Vision daily URL требует точно этот формат.
        """
        path = build_reaction_zip_path(2025, 4, 1, str(tmp_path))
        assert os.path.basename(path) == "BTCUSDT-aggTrades-2025-04-01.zip"

    def test_day_zero_padded(self, tmp_path):
        """
        Контракт: однозначные day и month дополняются нулём.
        """
        path = build_reaction_zip_path(2025, 1, 9, str(tmp_path))
        assert "2025-01-09" in os.path.basename(path)

    def test_path_inside_base_dir(self, tmp_path):
        path = build_reaction_zip_path(2025, 4, 1, str(tmp_path))
        assert path.startswith(str(tmp_path))

    def test_does_not_create_file(self, tmp_path):
        """
        Контракт: чистая функция — никаких сайд-эффектов.
        """
        build_reaction_zip_path(2025, 4, 1, str(tmp_path))
        zip_path = os.path.join(str(tmp_path), "BTCUSDT-aggTrades-2025-04-01.zip")
        assert not os.path.exists(zip_path)

    def test_different_days_different_paths(self, tmp_path):
        p1 = build_reaction_zip_path(2025, 4, 1, str(tmp_path))
        p2 = build_reaction_zip_path(2025, 4, 2, str(tmp_path))
        assert p1 != p2


class TestDownloadReactionMonth:
    """
    Контракт download_reaction_month(year, month, day, out_dir, base_url) -> str | None:
    - Скачивает daily ZIP если файл не существует → возвращает путь.
    - Idempotent: файл уже есть → пропускает скачивание, возвращает путь.
    - При ошибке сети → возвращает None (не поднимает исключение).
    - Создаёт out_dir если не существует.
    - URL: {base_url}/BTCUSDT-aggTrades-YYYY-MM-DD.zip

    WHY отдельная функция: daily ZIP (~22 MB каждый) vs monthly (~660 MB).
    Аналогична download_anchor_month — тот же контракт, другой URL-шаблон.
    """

    BASE_URL = (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT"
    )

    def _make_fake_zip(self, path: str) -> None:
        with zipfile.ZipFile(path, 'w') as zf:
            zf.writestr("dummy.csv", "col1,col2\n1,2\n")

    def test_returns_path_on_success(self, tmp_path):
        """
        Контракт: успешное скачивание → возвращает строку-путь к файлу.
        """
        def fake_urlretrieve(url, dest, reporthook=None):
            self._make_fake_zip(dest)

        with patch("download_anchor_data.urllib.request.urlretrieve",
                   side_effect=fake_urlretrieve):
            result = download_reaction_month(2025, 4, 1, str(tmp_path), self.BASE_URL)

        assert isinstance(result, str)
        assert result.endswith(".zip")

    def test_file_created_on_disk(self, tmp_path):
        """
        Контракт: после успешного скачивания файл существует на диске.
        """
        def fake_urlretrieve(url, dest, reporthook=None):
            self._make_fake_zip(dest)

        with patch("download_anchor_data.urllib.request.urlretrieve",
                   side_effect=fake_urlretrieve):
            result = download_reaction_month(2025, 4, 1, str(tmp_path), self.BASE_URL)

        assert os.path.exists(result)

    def test_skip_if_file_exists(self, tmp_path):
        """
        Контракт: файл уже существует → urlretrieve НЕ вызывается (idempotent).
        """
        zip_path = os.path.join(str(tmp_path), "BTCUSDT-aggTrades-2025-04-01.zip")
        self._make_fake_zip(zip_path)

        with patch("download_anchor_data.urllib.request.urlretrieve") as mock_dl:
            result = download_reaction_month(2025, 4, 1, str(tmp_path), self.BASE_URL)

        mock_dl.assert_not_called()
        assert result == zip_path

    def test_returns_none_on_network_error(self, tmp_path):
        """
        Контракт: ошибка сети → None, не исключение.
        """
        with patch(
            "download_anchor_data.urllib.request.urlretrieve",
            side_effect=Exception("Network error"),
        ):
            result = download_reaction_month(2025, 4, 1, str(tmp_path), self.BASE_URL)

        assert result is None

    def test_incomplete_file_removed_on_error(self, tmp_path):
        """
        Контракт: при ошибке неполный файл удаляется.
        WHY: аналогично download_anchor_month — неполный ZIP сломает pipeline.
        """
        zip_path = os.path.join(str(tmp_path), "BTCUSDT-aggTrades-2025-04-01.zip")

        def fail_after_partial(url, dest, reporthook=None):
            with open(dest, 'w') as f:
                f.write("partial")
            raise Exception("Connection lost")

        with patch("download_anchor_data.urllib.request.urlretrieve",
                   side_effect=fail_after_partial):
            download_reaction_month(2025, 4, 1, str(tmp_path), self.BASE_URL)

        assert not os.path.exists(zip_path)

    def test_creates_out_dir_if_not_exists(self, tmp_path):
        """
        Контракт: out_dir создаётся автоматически.
        """
        new_dir = os.path.join(str(tmp_path), "daily", "reaction")
        assert not os.path.exists(new_dir)

        def fake_urlretrieve(url, dest, reporthook=None):
            self._make_fake_zip(dest)

        with patch("download_anchor_data.urllib.request.urlretrieve",
                   side_effect=fake_urlretrieve):
            download_reaction_month(2025, 4, 1, new_dir, self.BASE_URL)

        assert os.path.isdir(new_dir)

    def test_url_constructed_correctly(self, tmp_path):
        """
        Контракт: URL = {base_url}/BTCUSDT-aggTrades-YYYY-MM-DD.zip.
        WHY: daily формат отличается от monthly — три части даты, не две.
        """
        captured_url = []

        def capture_url(url, dest, reporthook=None):
            captured_url.append(url)
            self._make_fake_zip(dest)

        with patch("download_anchor_data.urllib.request.urlretrieve",
                   side_effect=capture_url):
            download_reaction_month(2025, 4, 1, str(tmp_path), self.BASE_URL)

        assert len(captured_url) == 1
        assert captured_url[0] == f"{self.BASE_URL}/BTCUSDT-aggTrades-2025-04-01.zip"


# ---------------------------------------------------------------------------
# Этап 9: daily klines (1m) — build_klines_zip_path + download_klines_day
# ---------------------------------------------------------------------------

from download_anchor_data import build_klines_zip_path, download_klines_day


class TestBuildKlinesZipPath:
    """
    Контракт build_klines_zip_path(year, month, day, base_dir) -> str:
    - Имя файла: BTCUSDT-1m-YYYY-MM-DD.zip
    - Путь вложен в base_dir.
    - Чистая функция — не создаёт файл/папку.
    """

    def test_returns_string(self, tmp_path):
        result = build_klines_zip_path(2025, 4, 1, str(tmp_path))
        assert isinstance(result, str)

    def test_filename_format(self, tmp_path):
        """
        Контракт: имя файла BTCUSDT-1m-YYYY-MM-DD.zip с ведущими нулями.
        WHY: Binance Vision klines URL требует точно этот формат.
        """
        path = build_klines_zip_path(2025, 4, 1, str(tmp_path))
        assert os.path.basename(path) == "BTCUSDT-1m-2025-04-01.zip"

    def test_day_and_month_zero_padded(self, tmp_path):
        """
        Контракт: однозначные day и month дополняются нулём.
        """
        path = build_klines_zip_path(2025, 1, 9, str(tmp_path))
        assert "2025-01-09" in os.path.basename(path)

    def test_path_inside_base_dir(self, tmp_path):
        path = build_klines_zip_path(2025, 4, 1, str(tmp_path))
        assert path.startswith(str(tmp_path))

    def test_does_not_create_file(self, tmp_path):
        """
        Контракт: чистая функция — никаких сайд-эффектов.
        """
        build_klines_zip_path(2025, 4, 1, str(tmp_path))
        zip_path = os.path.join(str(tmp_path), "BTCUSDT-1m-2025-04-01.zip")
        assert not os.path.exists(zip_path)

    def test_different_days_different_paths(self, tmp_path):
        p1 = build_klines_zip_path(2025, 4, 1, str(tmp_path))
        p2 = build_klines_zip_path(2025, 4, 2, str(tmp_path))
        assert p1 != p2


class TestDownloadKlinesDay:
    """
    Контракт download_klines_day(year, month, day, out_dir, base_url) -> str | None:
    - Скачивает daily klines ZIP → возвращает путь.
    - Idempotent: файл уже есть → пропускает скачивание, возвращает путь.
    - При ошибке сети → возвращает None (не поднимает исключение).
    - Создаёт out_dir если не существует.
    - URL: {base_url}/BTCUSDT-1m-YYYY-MM-DD.zip
    """

    BASE_URL = (
        "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m"
    )

    def _make_fake_zip(self, path: str) -> None:
        with zipfile.ZipFile(path, 'w') as zf:
            zf.writestr("dummy.csv", "col1,col2\n1,2\n")

    def test_returns_path_on_success(self, tmp_path):
        """
        Контракт: успешное скачивание → возвращает строку-путь к файлу.
        """
        def fake_urlretrieve(url, dest, reporthook=None):
            self._make_fake_zip(dest)

        with patch("download_anchor_data.urllib.request.urlretrieve",
                   side_effect=fake_urlretrieve):
            result = download_klines_day(2025, 4, 1, str(tmp_path), self.BASE_URL)

        assert isinstance(result, str)
        assert result.endswith(".zip")

    def test_skip_if_file_exists(self, tmp_path):
        """
        Контракт: файл уже существует → urlretrieve НЕ вызывается (idempotent).
        """
        zip_path = os.path.join(str(tmp_path), "BTCUSDT-1m-2025-04-01.zip")
        self._make_fake_zip(zip_path)

        with patch("download_anchor_data.urllib.request.urlretrieve") as mock_dl:
            result = download_klines_day(2025, 4, 1, str(tmp_path), self.BASE_URL)

        mock_dl.assert_not_called()
        assert result == zip_path

    def test_returns_none_on_network_error(self, tmp_path):
        """
        Контракт: ошибка сети → None, не исключение.
        """
        with patch(
            "download_anchor_data.urllib.request.urlretrieve",
            side_effect=Exception("Network error"),
        ):
            result = download_klines_day(2025, 4, 1, str(tmp_path), self.BASE_URL)

        assert result is None

    def test_url_constructed_correctly(self, tmp_path):
        """
        Контракт: URL = {base_url}/BTCUSDT-1m-YYYY-MM-DD.zip.
        WHY: klines URL содержит '1m' в имени файла — отличается от aggTrades.
        """
        captured_url = []

        def capture_url(url, dest, reporthook=None):
            captured_url.append(url)
            self._make_fake_zip(dest)

        with patch("download_anchor_data.urllib.request.urlretrieve",
                   side_effect=capture_url):
            download_klines_day(2025, 4, 1, str(tmp_path), self.BASE_URL)

        assert len(captured_url) == 1
        assert captured_url[0] == f"{self.BASE_URL}/BTCUSDT-1m-2025-04-01.zip"
