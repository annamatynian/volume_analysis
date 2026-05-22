"""
onchain_cache.py
================
Локальный parquet-кэш для BGeometrics API (bitcoin-data.com).

Проблема: бесплатный тариф = 8 запросов/час.
Решение: кэшируем ответы в parquet, дёргаем API только если кэш старше max_age_hours.

Дефолт: 23 часа (не 24 — буфер против граничных случаев смены часового окна).

Fallback: если API падает с исключением (429, сеть) — возвращаем устаревший кэш
если он есть. Лучше старые данные, чем ничего.

Использование:
    from onchain_cache import CachedBGeometricsClient
    from onchain_client import BGeometricsClient

    inner = BGeometricsClient()
    client = CachedBGeometricsClient(inner, cache_dir=Path("data/onchain_cache"))

    df = await client.get_realized_loss_lth_usd()  # API или кэш — прозрачно
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

# WHY __file__: абсолютный путь к модулю — не зависит от cwd при запуске.
# Все данные гарантированно пишутся на D:, а не в произвольный cwd.
_MODULE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CACHE_DIR = _MODULE_DIR / "data" / "onchain_cache"

import pandas as pd

# WHY: onchain_client импортируется только для type hints → нет circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from onchain_client import BGeometricsClient


# ---------------------------------------------------------------------------
# Чистые функции — тестируются изолированно
# ---------------------------------------------------------------------------

def build_cache_path(metric: str, cache_dir: Union[str, Path]) -> Path:
    """
    Возвращает путь к parquet-файлу кэша для данной метрики.

    Args:
        metric:    Имя метрики, например 'sth-realized-price'.
        cache_dir: Директория для хранения кэша.

    Returns:
        Path к файлу, например: cache_dir/sth-realized-price.parquet
    """
    safe_name = metric.replace("/", "_").replace("\\", "_")
    return Path(cache_dir) / f"{safe_name}.parquet"


def is_cache_fresh(path: Union[str, Path], max_age_hours: float = 23) -> bool:
    """
    True если файл существует и его mtime моложе max_age_hours.

    Args:
        path:          Путь к файлу кэша.
        max_age_hours: Максимальный возраст в часах (дефолт: 23).

    Returns:
        True → кэш свежий, API запрашивать не нужно.
        False → кэша нет или устарел.
    """
    p = Path(path)
    if not p.exists():
        return False

    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    age = datetime.now() - mtime
    return age < timedelta(hours=max_age_hours)


def load_from_cache(path: Union[str, Path]) -> pd.DataFrame:
    """
    Загрузить DataFrame из parquet-файла.

    Args:
        path: Путь к parquet-файлу.

    Returns:
        DataFrame с данными из кэша.
    """
    return pd.read_parquet(Path(path))


def save_to_cache(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Сохранить DataFrame в parquet-файл.
    Создаёт родительские директории если нужно.

    Args:
        df:   DataFrame для сохранения.
        path: Путь к целевому файлу.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


# ---------------------------------------------------------------------------
# CachedBGeometricsClient — обёртка с cache-first логикой
# ---------------------------------------------------------------------------

class CachedBGeometricsClient:
    """
    Обёртка над BGeometricsClient с локальным parquet-кэшем.

    Логика для каждого метода:
      1. Проверяем кэш (is_cache_fresh).
      2. Свежий → возвращаем из файла, API не трогаем.
      3. Нет/старый → вызываем API → сохраняем в кэш → возвращаем.
      4. API падает + есть старый кэш → fallback на кэш (лучше старые данные).
      5. API падает + кэша нет → пробрасываем исключение.

    Args:
        client:        Внутренний BGeometricsClient (реальный или мок).
        cache_dir:     Директория для хранения parquet-файлов.
        max_age_hours: Возраст кэша при котором он считается свежим (дефолт: 23ч).
    """

    def __init__(
        self,
        client: "BGeometricsClient",
        cache_dir: Union[str, Path] = _DEFAULT_CACHE_DIR,
        max_age_hours: float = 23,
    ) -> None:
        self._client = client
        self._cache_dir = Path(cache_dir)
        self._max_age_hours = max_age_hours

    async def _get_with_cache(
        self,
        metric: str,
        api_coro,
    ) -> pd.DataFrame:
        """
        Универсальный cache-first getter.

        Args:
            metric:   Имя метрики (ключ кэша).
            api_coro: Корутина без аргументов → pd.DataFrame.

        Returns:
            DataFrame из кэша или от API.
        """
        cache_path = build_cache_path(metric, self._cache_dir)
        has_stale = cache_path.exists()

        # 1. Свежий кэш → возвращаем сразу
        if is_cache_fresh(cache_path, self._max_age_hours):
            return load_from_cache(cache_path)

        # 2. Нет свежего кэша → вызываем API
        try:
            df = await api_coro()
            save_to_cache(df, cache_path)
            return df
        except Exception as _cache_exc:
            # 3. API упал → fallback на устаревший кэш если есть
            if has_stale:
                _age_h = (
                    datetime.now()
                    - datetime.fromtimestamp(cache_path.stat().st_mtime)
                ).total_seconds() / 3600
                print(
                    f'[WARN] {metric}: API недоступен ({_cache_exc.__class__.__name__}: {_cache_exc}), '
                    f'используется устаревший кэш ({_age_h:.0f}ч назад).'
                )
                return load_from_cache(cache_path)
            raise  # кэша нет → пробрасываем исключение

    async def get_realized_loss_lth_usd(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """
        Суточный реализованный убыток LTH в USD. Cache-first.

        Returns:
            DataFrame [date, realized_loss_usd]
        """
        return await self._get_with_cache(
            metric="realized-loss-lth-usd",
            api_coro=lambda: self._client.get_realized_loss_lth_usd(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_sth_realized_price(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """
        STH Realized Price. Cache-first.

        Returns:
            DataFrame [date, sth_realized_price]
        """
        return await self._get_with_cache(
            metric="sth-realized-price",
            api_coro=lambda: self._client.get_sth_realized_price(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_utxos_in_loss_count(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """
        Количество UTXO в убытке. Cache-first.

        Returns:
            DataFrame [date, utxos_in_loss]
        """
        return await self._get_with_cache(
            metric="utxos-in-loss-count",
            api_coro=lambda: self._client.get_utxos_in_loss_count(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    # ------------------------------------------------------------------
    # Holder Structure — 5 новых методов (сессия 2026-05-10)
    # ------------------------------------------------------------------

    async def get_lth_mvrv(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """
        LTH MVRV ratio. Cache-first.

        Returns:
            DataFrame [date, lth_mvrv]
        """
        return await self._get_with_cache(
            metric="lth-mvrv",
            api_coro=lambda: self._client.get_lth_mvrv(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_sth_mvrv(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """
        STH MVRV ratio. Cache-first.

        Returns:
            DataFrame [date, sth_mvrv]
        """
        return await self._get_with_cache(
            metric="sth-mvrv",
            api_coro=lambda: self._client.get_sth_mvrv(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_lth_sopr(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """
        LTH SOPR. Cache-first.

        Returns:
            DataFrame [date, lth_sopr]
        """
        return await self._get_with_cache(
            metric="lth-sopr",
            api_coro=lambda: self._client.get_lth_sopr(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_lth_net_position_change_30d(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """
        LTH Net Position Change 30d (BTC). Cache-first.

        Returns:
            DataFrame [date, lth_net_position_30d]
        """
        return await self._get_with_cache(
            metric="lth-net-position-change-30d-btc",
            api_coro=lambda: self._client.get_lth_net_position_change_30d(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_sth_net_position_change_30d(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """
        STH Net Position Change 30d (BTC). Cache-first.

        Returns:
            DataFrame [date, sth_net_position_30d]
        """
        return await self._get_with_cache(
            metric="sth-net-position-change-30d-btc",
            api_coro=lambda: self._client.get_sth_net_position_change_30d(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    # ------------------------------------------------------------------
    # P1–P5: Expansion — 6 новых методов (сессия 2026-05-11)
    # ------------------------------------------------------------------

    async def get_sth_sopr(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """P1: STH SOPR. Cache-first.

        Returns:
            DataFrame [date, sth_sopr]
        """
        return await self._get_with_cache(
            metric="sth-sopr",
            api_coro=lambda: self._client.get_sth_sopr(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_exchange_netflow(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """P2: Exchange Net Flow BTC. Cache-first.

        Returns:
            DataFrame [date, exchange_netflow_btc]
        """
        return await self._get_with_cache(
            metric="exchange-netflow-btc",
            api_coro=lambda: self._client.get_exchange_netflow(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_nupl_lth(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """P3: LTH NUPL. Cache-first.

        Returns:
            DataFrame [date, nupl_lth]
        """
        return await self._get_with_cache(
            metric="nupl-lth",
            api_coro=lambda: self._client.get_nupl_lth(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_nupl_sth(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """P3: STH NUPL. Cache-first.

        Returns:
            DataFrame [date, nupl_sth]
        """
        return await self._get_with_cache(
            metric="nupl-sth",
            api_coro=lambda: self._client.get_nupl_sth(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_etf_flow(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """P4: ETF Flow BTC. Cache-first.

        Returns:
            DataFrame [date, etf_flow_btc]
        """
        return await self._get_with_cache(
            metric="etf-flow-btc",
            api_coro=lambda: self._client.get_etf_flow(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_hodl_waves(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """P5: HODL Waves Supply. Cache-first.

        Returns:
            DataFrame [date, <когортные колонки>]
        """
        return await self._get_with_cache(
            metric="hodl-waves-supply",
            api_coro=lambda: self._client.get_hodl_waves(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    # ------------------------------------------------------------------
    # МБ-01..08: Expansion — 8 методов (сессия 2026-05-21)
    # WHY отсутствовали: добавлены в BGeometricsClient в сессии 2026-05-20,
    # но не перенесены в CachedBGeometricsClient → краш [HOLDER STRUCTURE].
    # ------------------------------------------------------------------

    async def get_realized_price(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """МБ-01: Realized Price (средняя цена покупки всех BTC). Cache-first.

        Returns:
            DataFrame [date, realized_price]
        """
        return await self._get_with_cache(
            metric="realized-price",
            api_coro=lambda: self._client.get_realized_price(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_true_market_mean(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """МБ-02: True Market Mean (рубикон медвежьего рынка). Cache-first.

        Returns:
            DataFrame [date, true_market_mean]
        """
        return await self._get_with_cache(
            metric="true-market-mean",
            api_coro=lambda: self._client.get_true_market_mean(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_utxos_in_profit_pct(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """МБ-03: % UTXOs в прибыли (аппроксимация % STH в прибыли). Cache-first.

        Returns:
            DataFrame [date, utxos_in_profit_pct]
        """
        return await self._get_with_cache(
            metric="utxos-in-profit-pct",
            api_coro=lambda: self._client.get_utxos_in_profit_pct(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_supply_loss(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """МБ-04: Supply in Loss (BTC в нереализованном убытке). Cache-first.

        Returns:
            DataFrame [date, supply_loss]
        """
        return await self._get_with_cache(
            metric="supply-loss",
            api_coro=lambda: self._client.get_supply_loss(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_lth_realized_profit_usd(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """МБ-05: LTH Realized Profit USD (дневная прибыль LTH). Cache-first.

        Returns:
            DataFrame [date, lth_realized_profit_usd]
        """
        return await self._get_with_cache(
            metric="realized-profit-lth-usd",
            api_coro=lambda: self._client.get_lth_realized_profit_usd(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_nupl(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """МБ-06: NUPL (Net Unrealized Profit/Loss всего рынка). Cache-first.

        Returns:
            DataFrame [date, nupl]
        """
        return await self._get_with_cache(
            metric="nupl",
            api_coro=lambda: self._client.get_nupl(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_mvrv_zscore(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """МБ-07: MVRV Z-Score (макро позиционирование). Cache-first.

        Returns:
            DataFrame [date, mvrv_zscore]
        """
        return await self._get_with_cache(
            metric="mvrv-zscore",
            api_coro=lambda: self._client.get_mvrv_zscore(
                start_date=start_date,
                end_date=end_date,
            ),
        )

    async def get_realized_cap_hodl_waves(
        self,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """МБ-08: Realized Cap HODL Waves (RC-доли по возрасту UTXO). Cache-first.

        Returns:
            DataFrame [date, age_0d_1d, age_1d_1w, …, age_10y]
        """
        return await self._get_with_cache(
            metric="realized-cap-hodl-waves",
            api_coro=lambda: self._client.get_realized_cap_hodl_waves(
                start_date=start_date,
                end_date=end_date,
            ),
        )
