# tests/test_mozart_nupl.py
# МБ-06 | Общий NUPL — 30-дн MA, зоны Mozart (пост 15.05.2026)
#
# Endpoint: /v1/nupl → поле nupl (float, десятичная дробь; 0.3004 = 30.04%)
# Диагностика 20.05.2026: 200 OK, 1460 записей, формат dict.
#
# Зоны:
#   BULL        : nupl >= bull_threshold (0.50) — эйфория
#   HOPE        : hope_threshold (0.25) <= nupl < bull_threshold
#   EARLY_BEAR  : 0.0 <= nupl < hope_threshold — Mozart's пик ~18% сюда
#   BEAR        : bottom_target < nupl < 0.0 — медвежий рынок
#   BOTTOM_ZONE : nupl <= bottom_target (-0.40) — целевое дно Mozart

import pytest
from mozart_config import MOZART_CONFIG
from mozart_signals import classify_nupl_regime


# ---------------------------------------------------------------------------
# Контракт конфига — ключи и типы
# ---------------------------------------------------------------------------

class TestNuplConfig:
    def test_nupl_bull_threshold_exists(self):
        assert "nupl_bull_threshold" in MOZART_CONFIG
        # WHY: classify_nupl_regime читает этот ключ напрямую;
        #   отсутствие → KeyError в production при каждом запуске оркестратора.

    def test_nupl_hope_threshold_exists(self):
        assert "nupl_hope_threshold" in MOZART_CONFIG
        # WHY: граница HOPE/EARLY_BEAR; отсутствие → KeyError при классификации.

    def test_nupl_bottom_target_exists(self):
        assert "nupl_bottom_target" in MOZART_CONFIG
        # WHY: цель дна по Mozart (-40%); отсутствие → BOTTOM_ZONE недостижима,
        #   оркестратор никогда не сигнализирует исторический уровень.

    def test_thresholds_are_numeric(self):
        for key in ("nupl_bull_threshold", "nupl_hope_threshold", "nupl_bottom_target"):
            assert isinstance(float(MOZART_CONFIG[key]), float)
            # WHY: classify_nupl_regime использует float-сравнение;
            #   str-значение в конфиге даст TypeError или неверный результат.

    def test_nupl_bottom_target_is_negative(self):
        assert float(MOZART_CONFIG["nupl_bottom_target"]) < 0
        # WHY: BOTTOM_ZONE = нереализованный убыток = NUPL отрицательный.
        #   Положительное значение → BOTTOM_ZONE никогда не будет достигнута.

    def test_thresholds_ordered(self):
        bull   = float(MOZART_CONFIG["nupl_bull_threshold"])
        hope   = float(MOZART_CONFIG["nupl_hope_threshold"])
        bottom = float(MOZART_CONFIG["nupl_bottom_target"])
        assert bottom < 0.0 < hope < bull
        # WHY: нарушение порядка → зоны перекрываются или инвертируются;
        #   сигналы оркестратора становятся непредсказуемы при изменении конфига.


# ---------------------------------------------------------------------------
# Контракт типа возврата
# ---------------------------------------------------------------------------

class TestNuplReturnType:
    def test_returns_str(self):
        result = classify_nupl_regime(0.0)
        assert isinstance(result, str)
        # WHY: оркестратор встраивает результат в f-строку блока;
        #   не-str → TypeError при форматировании вывода.


# ---------------------------------------------------------------------------
# Зоны — типичные значения (центры диапазонов)
# ---------------------------------------------------------------------------

class TestClassifyNuplRegimeZones:
    def test_bull_zone(self):
        bull = float(MOZART_CONFIG["nupl_bull_threshold"])
        result = classify_nupl_regime(bull + 0.10)
        assert result == "BULL"
        # WHY: выше bull_threshold = эйфория; оркестратор должен сигнализировать
        #   риск распределения. Неверный лейбл → пропуск сигнала в [FINAL VERDICT].

    def test_hope_zone(self):
        bull = float(MOZART_CONFIG["nupl_bull_threshold"])
        hope = float(MOZART_CONFIG["nupl_hope_threshold"])
        mid  = (bull + hope) / 2
        result = classify_nupl_regime(mid)
        assert result == "HOPE"
        # WHY: текущее API значение 0.3004 (19.05.2026) попадает в HOPE.
        #   Неверный лейбл → оркестратор неверно оценивает текущую фазу рынка.

    def test_early_bear_zone(self):
        hope = float(MOZART_CONFIG["nupl_hope_threshold"])
        mid  = hope / 2  # середина [0, hope), напр. 0.125
        result = classify_nupl_regime(mid)
        assert result == "EARLY_BEAR"
        # WHY: Mozart's пик марта 2026 ~0.18 попадает сюда (0.0 <= 0.18 < 0.25).
        #   Неверный лейбл → потеря контекста «ранняя медвежка» при историческом
        #   сравнении в оркестраторе.

    def test_bear_zone(self):
        bottom = float(MOZART_CONFIG["nupl_bottom_target"])
        mid    = bottom / 2  # середина (bottom, 0), напр. -0.20
        result = classify_nupl_regime(mid)
        assert result == "BEAR"
        # WHY: отрицательный NUPL выше цели = медвежий рынок без дна.
        #   Неверный лейбл → оркестратор не отличает ранний убыток от дна цикла.

    def test_bottom_zone(self):
        bottom = float(MOZART_CONFIG["nupl_bottom_target"])
        result = classify_nupl_regime(bottom - 0.05)
        assert result == "BOTTOM_ZONE"
        # WHY: Mozart: ~40% нереализованных убытков = целевое дно текущего цикла.
        #   Неверный лейбл → оркестратор не сигнализирует достижение исторического
        #   целевого уровня, пропуск ключевого сигнала в [FINAL VERDICT].


# ---------------------------------------------------------------------------
# Граничные значения
# ---------------------------------------------------------------------------

class TestClassifyNuplRegimeBoundaries:
    def test_bull_threshold_exact_is_bull(self):
        """Ровно на bull_threshold → BULL (включительно)."""
        threshold = float(MOZART_CONFIG["nupl_bull_threshold"])
        result = classify_nupl_regime(threshold)
        assert result == "BULL"
        # WHY: граница включительно = достижение порога активирует зону.
        #   Строгий > без включительно → nupl ровно 0.50 падало бы в HOPE;
        #   тихая потеря сигнала при точном попадании в порог.

    def test_hope_threshold_exact_is_hope(self):
        """Ровно на hope_threshold → HOPE (нижняя граница включительно)."""
        threshold = float(MOZART_CONFIG["nupl_hope_threshold"])
        result = classify_nupl_regime(threshold)
        assert result == "HOPE"
        # WHY: hope_threshold — нижняя граница HOPE.
        #   Строгий > → nupl = 0.25 стало бы EARLY_BEAR; тихий зональный сдвиг
        #   на значении ровно на пороге.

    def test_zero_is_early_bear(self):
        """NUPL == 0.0 → EARLY_BEAR (рубикон нуля = нижняя граница EARLY_BEAR)."""
        result = classify_nupl_regime(0.0)
        assert result == "EARLY_BEAR"
        # WHY: NUPL = 0 = рынок ровно на безубытке, нет убытка когортно.
        #   Переход в BEAR начинается при nupl < 0.
        #   Если 0.0 → BEAR: оркестратор показывает ложную капитуляцию при
        #   нейтральном рынке.

    def test_just_above_zero_is_early_bear(self):
        """Минимально выше 0 → EARLY_BEAR, не BEAR."""
        result = classify_nupl_regime(0.001)
        assert result == "EARLY_BEAR"
        # WHY: любой положительный NUPL ниже hope_threshold = ранняя медвежка.
        #   nupl = 0.001 → BEAR = неверный сигнал убытков при минимальной прибыли.

    def test_just_below_zero_is_bear(self):
        """Минимально ниже 0 → BEAR, не EARLY_BEAR."""
        result = classify_nupl_regime(-0.001)
        assert result == "BEAR"
        # WHY: пересечение нуля вниз = рынок стал убыточным когортно = BEAR.
        #   nupl = -0.001 → EARLY_BEAR = скрытый убыточный рынок классифицирован
        #   как ещё-не-медвежий; пропуск смены режима.

    def test_bottom_target_exact_is_bottom_zone(self):
        """Ровно на bottom_target → BOTTOM_ZONE (включительно)."""
        target = float(MOZART_CONFIG["nupl_bottom_target"])
        result = classify_nupl_regime(target)
        assert result == "BOTTOM_ZONE"
        # WHY: Mozart's цель (-40%) включительно = достижение уровня = сигнал дна.
        #   Строгий < без включительно → nupl ровно -0.40 оставалось бы в BEAR;
        #   оркестратор не сигнализировал бы достижение исторической цели цикла.

    def test_extreme_negative_is_bottom_zone(self):
        """Глубоко отрицательный NUPL → BOTTOM_ZONE (ниже цели)."""
        result = classify_nupl_regime(-0.99)
        assert result == "BOTTOM_ZONE"
        # WHY: устойчивость к экстремальным API-значениям; функция не должна
        #   падать или возвращать неожиданный лейбл при нестандартных данных.
