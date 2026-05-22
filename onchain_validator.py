import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from onchain_client import BGeometricsClient


@dataclass
class ValidationResult:
    price_level: float
    is_valid: bool
    z_score: float
    urpd_thickness: float  # % от total supply
    mean_reversion_probability: float
    reason: str

class OnChainValidator:
    """
    Валидация Volume Profile nodes через on-chain cost basis
    """
    
    def __init__(self, client: BGeometricsClient):
        self.client = client
        self.sth_historical: Optional[pd.DataFrame] = None
    
    async def initialize(self, lookback_days: int = 365):
        """
        Загрузить исторические STH Realized Price для Z-score расчётов
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        self.sth_historical = await self.client.get_sth_realized_price(
            start_date=start_date,
            end_date=end_date
        )
    
    def calculate_z_score(self, current_price: float) -> float:
        """
        Z-score отклонения текущей цены от STH cost basis
        
        Интерпретация:
        - Z < -2: Цена НИЖЕ cost basis (capitulation zone)
        - Z > +2: Цена ВЫШЕ cost basis (euphoria zone)
        """
        if self.sth_historical is None:
            raise RuntimeError("Call initialize() first")
        
        sth_mean = self.sth_historical['sth_realized_price'].mean()
        sth_std = self.sth_historical['sth_realized_price'].std()
        
        z_score = (current_price - sth_mean) / sth_std
        return z_score
    
    async def validate_vp_node(
        self,
        price_level: float,
        volume_profile_volume: float,  # Ваш текущий VP volume
        current_btc_price: float
    ) -> ValidationResult:
        """
        Валидация Volume Profile node через on-chain data
        
        Logic:
        1. Z-score: Насколько цена отклоняется от STH cost basis
        2. URPD: Сколько BTC было куплено на этом уровне
        3. Mean Reversion Probability: Комбинация обоих
        """
        
        # 1. Z-score validation
        z_score = self.calculate_z_score(price_level)
        
        # 2. URPD thickness
        urpd_data = await self.client.get_urpd(price_level)
        urpd_thickness = urpd_data['percent_supply']
        
        # 3. Mean Reversion Probability (эвристика)
        # High URPD + Low Z-score = Strong support
        if urpd_thickness > 2.0 and z_score < -1.0:
            mean_reversion_prob = 0.75
            is_valid = True
            reason = f"Strong support: {urpd_thickness:.2f}% supply, Z={z_score:.2f}"
        
        elif urpd_thickness > 1.0 and -2.0 < z_score < 0:
            mean_reversion_prob = 0.60
            is_valid = True
            reason = f"Moderate support: {urpd_thickness:.2f}% supply"
        
        else:
            mean_reversion_prob = 0.30
            is_valid = False
            reason = f"Weak support: URPD {urpd_thickness:.2f}%, Z={z_score:.2f}"
        
        return ValidationResult(
            price_level=price_level,
            is_valid=is_valid,
            z_score=z_score,
            urpd_thickness=urpd_thickness,
            mean_reversion_probability=mean_reversion_prob,
            reason=reason
        )

    def check_capitulation_signal(
        self,
        realized_loss_df: pd.DataFrame,
        threshold_usd: float = 300_000_000,
    ) -> bool:
        """
        True если последние 3 дня реализованный убыток LTH > threshold.
        Сигнал: структурный кризис разрешается, POC как цель актуален.

        Args:
            realized_loss_df: DataFrame с колонками [date, lth_realized_loss_usd].
            threshold_usd:    Порог в USD (дефолт $300M).

        Returns:
            True если все 3 последних значения > threshold, иначе False.
        """
        # WHY bool(): .all() возвращает numpy.bool_ — приводим к python bool для чистого контракта
        if realized_loss_df.empty:
            return False
        recent = realized_loss_df.tail(3)
        return bool((recent['lth_realized_loss_usd'] > threshold_usd).all())