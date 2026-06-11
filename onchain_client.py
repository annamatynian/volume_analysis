import requests
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd


class BGeometricsClient:
    """
    Client для BGeometrics API (bitcoin-data.com).

    WHY requests вместо aiohttp: aiohttp использует собственный async DNS
    resolver (aiodns/c-ares) который на Windows не резолвит bitcoin-data.com,
    тогда как системный DNS (urllib/requests) работает нормально.
    Для одного запроса при старте оркестратора синхронный клиент достаточен.
    """

    BASE_URL = "https://bitcoin-data.com/api/v1"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'volume_analysis/1.0'})

    async def close(self):
        # WHY no-op: requests.Session не требует явного закрытия.
        # Метод оставлен для обратной совместимости с оркестратором.
        self.session.close()

    # ------------------------------------------------------------------
    # Внутренний helper — единая точка HTTP + нормализация DataFrame
    # ------------------------------------------------------------------

    async def _fetch_timeseries(
        self,
        endpoint_slug: str,
        value_col: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Универсальный fetcher временного ряда BGeometrics.

        Args:
            endpoint_slug: Суффикс эндпоинта, напр. 'lth-mvrv'.
            value_col:     Имя колонки значения в результирующем DataFrame.
            start_date:    Начало диапазона (параметр 'start').
            end_date:      Конец диапазона (параметр 'end').

        Returns:
            DataFrame с колонками строго [date, value_col].
            date — datetime64, value_col — float64.

        WHY defensive rename: разные эндпоинты возвращают разные имена
        value-колонки (lthMvrv, lthSopr и т.д.) — берём первую не-date
        колонку и переименовываем в value_col. Это изолирует оркестратор
        от нестабильности API.
        WHY 'd' как primary: все известные BGeometrics-эндпоинты используют
        'd' для даты; fallback на первую колонку защищает от исключений.
        """
        url = f"{self.BASE_URL}/{endpoint_slug}"

        params = {}
        if start_date:
            params['start'] = start_date.strftime('%Y-%m-%d')
        if end_date:
            params['end'] = end_date.strftime('%Y-%m-%d')

        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return pd.DataFrame(columns=['date', value_col])

        df = pd.DataFrame(data)

        # Определяем дата-колонку: 'd' приоритет, иначе первая колонка
        date_col_raw = 'd' if 'd' in df.columns else df.columns[0]
        # Value-колонка: первая колонка, отличная от дата-колонки и не 'unixTs'
        # WHY пропускаем unixTs: BGeometrics возвращает 3 поля [d, unixTs, значение]—
        # без явного исключения next() возьмёт unixTs вместо реального значения.
        _SKIP_COLS = {date_col_raw, 'unixTs'}
        val_col_raw = next(
            (c for c in df.columns if c not in _SKIP_COLS), None
        )

        rename = {date_col_raw: 'date'}
        if val_col_raw:
            rename[val_col_raw] = value_col

        df = df.rename(columns=rename)
        df['date'] = pd.to_datetime(df['date'])
        if value_col in df.columns:
            df[value_col] = df[value_col].astype(float)

        # Возвращаем строго две колонки — лишние поля API отбрасываем
        return df[['date', value_col]]

    # ------------------------------------------------------------------
    # Существующие методы (без изменений)
    # ------------------------------------------------------------------

    async def get_sth_realized_price(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        STH Realized Price (Short-Term Holder cost basis).

        Returns:
            DataFrame [date, sth_realized_price]
        """
        endpoint = f"{self.BASE_URL}/sth-realized-price"

        params = {}
        if start_date:
            params['start'] = start_date.strftime('%Y-%m-%d')
        if end_date:
            params['end'] = end_date.strftime('%Y-%m-%d')

        resp = self.session.get(endpoint, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data)
        df = df.rename(columns={'d': 'date', 'sthRealizedPrice': 'sth_realized_price'})
        df['date'] = pd.to_datetime(df['date'])
        df['sth_realized_price'] = df['sth_realized_price'].astype(float)
        return df[['date', 'sth_realized_price']]

    async def get_realized_loss_lth_usd(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Реализованный убыток LTH (USD-деноминация).

        Endpoint: /v1/realized_loss_lth → поле realizedLossLth (float).
        WHY этот endpoint, не realized_loss_lth_usd:
          Варианты _usd и _btc возвращают 404 на free tier (paywall).
          Базовый realized_loss_lth возвращает USD-значения без суффикса.
          Диагностировано CO-5 (2026-05-20).
        WHY выходная колонка lth_realized_loss_usd сохранена:
          Оркестратор и тесты ссылаются на это имя — менять не нужно.

        Returns:
            DataFrame [date, lth_realized_loss_usd]
              lth_realized_loss_usd — float, USD, знак минус = убыток.
        """
        return await self._fetch_timeseries(
            'realized_loss_lth', 'lth_realized_loss_usd',
            start_date, end_date,
        )

    # ------------------------------------------------------------------
    # Holder Structure — 5 методов (сессия 2026-05-10)
    # ------------------------------------------------------------------

    async def get_lth_mvrv(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        LTH MVRV (Market Value / Realized Value для Long-Term Holders).

        WHY: >1.0 = LTH в нереализованной прибыли (возможный сбыт);
             <1.0 = LTH в убытке (давление продаж снижено).

        Returns:
            DataFrame [date, lth_mvrv]
        """
        return await self._fetch_timeseries('lth-mvrv', 'lth_mvrv', start_date, end_date)

    async def get_sth_mvrv(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        STH MVRV (Market Value / Realized Value для Short-Term Holders).

        WHY: STH чувствительнее к цене — их MVRV < 1.0 сигнализирует
             о давлении продаж и возможной капитуляции.

        Returns:
            DataFrame [date, sth_mvrv]
        """
        return await self._fetch_timeseries('sth-mvrv', 'sth_mvrv', start_date, end_date)

    async def get_lth_sopr(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        LTH SOPR (Spent Output Profit Ratio для Long-Term Holders).

        WHY: >1.0 = LTH продают монеты выше себестоимости (фиксируют прибыль);
             <1.0 = LTH продают ниже себестоимости (капитуляция);
             =1.0 = LTH продают по себестоимости (зона поддержки/сопротивления).

        Returns:
            DataFrame [date, lth_sopr]
        """
        return await self._fetch_timeseries('lth-sopr', 'lth_sopr', start_date, end_date)

    async def get_lth_net_position_change_30d(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        LTH Net Position Change за 30 дней (в BTC).

        WHY: Скользящая разница притока и оттока BTC в LTH-когорту.
             Положительное значение = LTH накапливают (бычий сигнал);
             Отрицательное = LTH распродают (фаза распределения).

        Returns:
            DataFrame [date, lth_net_position_30d]
        """
        return await self._fetch_timeseries(
            'lth-net-position-change-30d-btc', 'lth_net_position_30d',
            start_date, end_date,
        )

    async def get_sth_net_position_change_30d(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        STH Net Position Change за 30 дней (в BTC).

        WHY: STH накапливают = новый спрос входит в рынок;
             STH продают = краткосрочники выходят (давление продаж).

        Returns:
            DataFrame [date, sth_net_position_30d]
        """
        return await self._fetch_timeseries(
            'sth-net-position-change-30d-btc', 'sth_net_position_30d',
            start_date, end_date,
        )

    # ------------------------------------------------------------------
    # P1–P5: Expansion — 6 новых методов (сессия 2026-05-11)
    # ------------------------------------------------------------------

    async def get_sth_sopr(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        P1: STH SOPR (Spent Output Profit Ratio для Short-Term Holders).

        WHY: замыкает SOPR-пару с LTH SOPR.
             >1.0 = STH продают в прибыль;
             <1.0 = капитуляция STH.

        Returns:
            DataFrame [date, sth_sopr]
        """
        return await self._fetch_timeseries('sth-sopr', 'sth_sopr', start_date, end_date)

    async def get_exchange_netflow(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        P2: Exchange Net Flow (BTC).

        WHY: приток BTC на биржи = давление продаж;
             отток = накопление вне бирж.
             Знак: + = приток, − = отток.

        Returns:
            DataFrame [date, exchange_netflow_btc]
        """
        return await self._fetch_timeseries(
            'exchange-netflow-btc', 'exchange_netflow_btc', start_date, end_date
        )

    async def get_nupl_lth(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        P3: LTH NUPL (Net Unrealized Profit/Loss для Long-Term Holders).

        WHY: точнее MVRV — учитывает реальные позиции.
             >0.75 = эйфория LTH (риск разворота);
             <0   = LTH в убытке (медвежьй рынок).

        Returns:
            DataFrame [date, nupl_lth]
        """
        return await self._fetch_timeseries('nupl-lth', 'nupl_lth', start_date, end_date)

    async def get_nupl_sth(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        P3: STH NUPL (Net Unrealized Profit/Loss для Short-Term Holders).

        WHY: STH NUPL < 0 = STH в убытке в целом — капитуляция.

        Returns:
            DataFrame [date, nupl_sth]
        """
        return await self._fetch_timeseries('nupl-sth', 'nupl_sth', start_date, end_date)

    async def get_etf_flow(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        P4: ETF Flow BTC (spot Bitcoin ETF потоки с 2024 г.).

        WHY: приток в ETF при тестировании POC =
             институциональная поддержка.
             Знак: + = приток, − = отток.

        Returns:
            DataFrame [date, etf_flow_btc]
        """
        return await self._fetch_timeseries('etf-flow-btc', 'etf_flow_btc', start_date, end_date)

    async def get_hodl_waves(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        P5: HODL Waves Supply (распределение BTC по возрасту UTXO).

        WHY: единственный источник MTH-когорты (1–6 мес.).
        Структура JSON уточняется после диагностики эндпоинта.

        WHY не через _fetch_timeseries: эндпоинт возвращает несколько
        когортных колонок — _fetch_timeseries выбирает только одну.
        Здесь возвращаем весь DataFrame с date + всеми колонками
        (кроме unixTs).

        Returns:
            DataFrame [date, <когортные колонки>]  — состав уточняется после диагностики
        """
        url = f"{self.BASE_URL}/hodl-waves-supply"
        params = {}
        if start_date:
            params['start'] = start_date.strftime('%Y-%m-%d')
        if end_date:
            params['end'] = end_date.strftime('%Y-%m-%d')

        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return pd.DataFrame(columns=['date'])

        df = pd.DataFrame(data)

        # Rename date column
        date_col = 'd' if 'd' in df.columns else df.columns[0]
        df = df.rename(columns={date_col: 'date'})
        df['date'] = pd.to_datetime(df['date'])

        # Убираем unixTs — не несёт аналитической ценности
        if 'unixTs' in df.columns:
            df = df.drop(columns=['unixTs'])

        # WHY float cast: API возвращает значения когорт как строки
        # ('1427793.16350000') — подтверждено диагностикой 2026-05-11.
        # Без castа df['age_1m_3m'] / 1_000_000 бросает TypeError.
        for _col in [c for c in df.columns if c != 'date']:
            try:
                df[_col] = df[_col].astype(float)
            except (ValueError, TypeError):
                pass  # оставляем как есть если колонка неконвертируема

        return df

    async def get_utxos_in_profit_pct(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        МБ-03 | % UTXOs в прибыли — аппроксимация Mozart's '% STH в прибыли'.

        Endpoint: /v1/utxos-in-profit-pct → поле utxosInProfitPct (float, 0–100%).

        WHY этот endpoint, не /v1/profit-loss:
          /v1/profit-loss возвращает ratio (~1.3–2.1), не процент.
          /v1/utxos-in-profit-pct возвращает % UTXOs в прибыли (0–100%),
          что соответствует диапазонам Mozart (51–59% / 69–76% / 85–95%).

        WHY не STH-specific:
          BGeometrics не предоставляет отдельный endpoint для % STH в прибыли
          на free tier. utxosInProfitPct — лучшая доступная аппроксимация.
          Диагностировано 2026-05-16: текущее значение 77.37% (EUPHORIA_APPROACH).

        WHY date params игнорируются API:
          Endpoint возвращает полную историю с 2022-12-07 независимо от
          параметров start/end. Для delta_7d берём последние 8+ записей.

        Returns:
            DataFrame [date, utxos_in_profit_pct] — хронологически по возрастанию.
        """
        return await self._fetch_timeseries(
            'utxos-in-profit-pct',
            'utxos_in_profit_pct',
            start_date,
            end_date,
        )

    async def get_realized_price(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        МБ-01 | Realized Price («Синяя линия» дна цикла).

        Endpoint: /v1/realized-price → поле realizedPrice (float, USD).

        WHY этот эндпоинт:
          Realized Price = средняя цена покупки всех BTC в обращении.
          Совпадает с дном циклов 2015, 2018, 2020.
          Mozart (пост 25.02.2026): «самый сильный уровень из трёхлинейной модели».
          Диагностика 20.05.2026: 200 OK, 1461 запись, поле realizedPrice.

        Returns:
            DataFrame [date, realized_price] — хронологически по возрастанию.
        """
        return await self._fetch_timeseries(
            'realized-price',
            'realized_price',
            start_date,
            end_date,
        )

    async def get_true_market_mean(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        МБ-02 | True Market Mean («Зелёная линия», рубикон медвежьего рынка).

        Endpoint: /v1/true-market-mean → поле trueMarketMean (float, USD).

        WHY этот эндпоинт:
          True Market Mean = Realized Price только по активным BTC в обращении.
          Mozart (пост 25.02.2026): «зелёная линия», пробой вниз =
          «обозначил смену глобального тренда». Рубикон медвежьего рынка.
          Диагностика 20.05.2026: 200 OK, 1458 записей, поле trueMarketMean.

        Returns:
            DataFrame [date, true_market_mean] — хронологически по возрастанию.
        """
        return await self._fetch_timeseries(
            'true-market-mean',
            'true_market_mean',
            start_date,
            end_date,
        )

    async def get_supply_loss(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        МБ-04 | Supply in Loss — объём BTC в нереализованном убытке.

        Endpoint: /v1/supply-loss → поле supplyLoss (float, BTC).

        WHY этот эндпоинт:
          Количество BTC, чья средняя цена покупки выше текущей.
          Mozart (02.04.2026, 08.04.2026): 5M = триггер смены структурного
          тренда; 3–3.5M = активное сопротивление для роста.
          Диагностика 20.05.2026: 200 OK, 1461 запись, поле supplyLoss.

        Returns:
            DataFrame [date, supply_loss] — хронологически по возрастанию.
        """
        return await self._fetch_timeseries(
            'supply-loss',
            'supply_loss',
            start_date,
            end_date,
        )

    async def get_nupl(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        МБ-06 | Общий NUPL — нереализованный P&L всего рынка.

        Endpoint: /v1/nupl → поле nupl (float, десятичная дробь).

        WHY этот эндпоинт:
          Net Unrealized Profit/Loss = (market cap - realized cap) / market cap.
          Положительный = рынок в совокупной прибыли; отрицательный = убыток.
          Mozart (пост 15.05.2026): целевое дно текущего цикла ~−0.40
          (40%-ая доля нереализованных убытков в среднем за месяц).
          Диагностика 20.05.2026: 200 OK, 1460 записей, поле nupl.

        Returns:
            DataFrame [date, nupl] — хронологически по возрастанию.
        """
        return await self._fetch_timeseries(
            'nupl',
            'nupl',
            start_date,
            end_date,
        )

    async def get_mvrv_zscore(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        МБ-07 | MVRV Z-Score — макро позиционирование относительно истории.

        Endpoint: /v1/mvrv-zscore → поле mvrvZscore (float).

        WHY этот эндпоинт:
          MVRV Z-Score = (market cap - realized cap) / std(market cap).
          Показывает насколько стандартных отклонений рынок находится
          выше/ниже средней цены покупки.
          Mozart (пост 25.02.2026): Z < 0 = рынок ниже Realized Price =
          зона исторического дна.
          Диагностика 20.05.2026: 200 OK, 1460 записей, поле mvrvZscore.

        Returns:
            DataFrame [date, mvrv_zscore] — хронологически по возрастанию.
        """
        return await self._fetch_timeseries(
            'mvrv-zscore',
            'mvrv_zscore',
            start_date,
            end_date,
        )

    async def get_realized_cap_hodl_waves(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        МБ-08 | Realized Cap HODL Waves — RC-доли по возрасту UTXO.

        Endpoint: /v1/realized-cap-hodl-waves
        Поля: age_0d_1d … age_10y (float, доли от общего realized cap, сумма ≈ 1.0).

        WHY не через _fetch_timeseries:
          Эндпоинт возвращает несколько когортных колонок — _fetch_timeseries
          выбирает только одну (первую не-дат-колонку).
          Аналогичная реализация get_hodl_waves() (по 2026-05-11).

        WHY этот эндпоинт:
          RC-доли (age_2y_3y, age_3y_4y) — числитель формулы OCA.
          Знаменатель — BTC supply из get_hodl_waves() (hodl-waves-supply).
          Диагностика 20.05.2026: 200 OK, 1455 записей,
          значения в виде строк ('0.056') — каст в float обязателен.

        Returns:
            DataFrame [date, age_0d_1d, age_1d_1w, …, age_10y]
            Все age_* колонки — float, доли от realized cap (0–1).
        """
        url = f"{self.BASE_URL}/realized-cap-hodl-waves"
        params = {}
        if start_date:
            params['start'] = start_date.strftime('%Y-%m-%d')
        if end_date:
            params['end'] = end_date.strftime('%Y-%m-%d')

        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return pd.DataFrame(columns=['date'])

        df = pd.DataFrame(data)

        date_col = 'd' if 'd' in df.columns else df.columns[0]
        df = df.rename(columns={date_col: 'date'})
        df['date'] = pd.to_datetime(df['date'])

        if 'unixTs' in df.columns:
            df = df.drop(columns=['unixTs'])

        # WHY float cast: API возвращает RC-доли как строки ('0.056').
        # Без каста вычисления RC-долей невозможны.
        for _col in [c for c in df.columns if c != 'date']:
            try:
                df[_col] = df[_col].astype(float)
            except (ValueError, TypeError):
                pass

        return df

    async def get_lth_realized_profit_usd(
        self,
        start_date=None,
        end_date=None,
    ):
        """
        МБ-05 | LTH Realized Profit USD — дневной прибыль LTH в USD.

        Endpoint: /v1/realized-profit-lth-usd → поле realizedProfitLthUsd (float, USD).

        WHY этот слуг, не realized_profit_lth_usd:
          Диагностика 2026-05-21:
            realized_profit_lth_usd   (подчёркивание) → 404
            realized_profit_lth       (подчёркивание) → 200 но BTC, не USD
            realized-profit-lth-usd   (дефисы)     → 200 OK, 1440 записей, USD ✓
            realized-profit-lth       (дефисы)     → 404
          USD необходим: Mozart сравнивает с порогом $1B/день.

        Returns:
            DataFrame [date, lth_realized_profit_usd]
              lth_realized_profit_usd — float, USD/день, всегда >= 0.
        """
        return await self._fetch_timeseries(
            'realized-profit-lth-usd',
            'lth_realized_profit_usd',
            start_date,
            end_date,
        )

    async def get_funding_rate_series(
        self,
        records: int = 90,
    ):
        """
        М-15 | Funding Rate 8h серия для расчёта 30d MA.

        Endpoint: /v1/funding-rate
        Поле: fundingRate (строка, доли; 0.0001 = базовая ставка Binance).
        Интервал: 8 часов (3 записи/день).

        WHY не через _fetch_timeseries:
          Диагностика 2026-06-02: параметр days игнорируется API —
          всегда возвращается вся история (3178 записей с 2023-07-09).
          Берём хвост (тайл последних `records` записей) без повторных запросов.

        Args:
            records: Количество 8-часовых записей для возврата.
                     30 дней × 3 записи/день = 90 (дефолт).
                     35 дней = 105 записей (если нужен запас).

        Returns:
            DataFrame [date, funding_rate]
              date         — datetime64, 8-часовые интервалы.
              funding_rate — float, доли (0.0001 = базовая ставка Binance).
        """
        url = f"{self.BASE_URL}/funding-rate"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:  # pragma: no cover
            return pd.DataFrame(columns=['date', 'funding_rate'])

        df = pd.DataFrame(data)
        df = df.rename(columns={'d': 'date'})
        # WHY format='mixed': API возвращает даты с миллисекундами ('2024-01-01 00:00:00.001').
        # Стандартный парсинг без формата вызывает UserWarning об остатке '.001'.
        # WHY floor('s'): обрезаем субсекундную часть — 8h-интервалы без миллисекунд.
        df['date'] = pd.to_datetime(df['date'], format='mixed').dt.floor('s')
        df['funding_rate'] = df['fundingRate'].astype(float)
        df = df[['date', 'funding_rate']].sort_values('date').reset_index(drop=True)

        # WHY tail: API игнорирует параметр days; берём хвост без повторного запроса.
        return df.tail(records).reset_index(drop=True)

