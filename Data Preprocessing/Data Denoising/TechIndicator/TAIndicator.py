import numpy as np
import talib


def DPO_gettatech(open, high, close, low, volumn, techlist=['EMA', 'MA']):
    ta_indicator = np.array([])
    for tech_name in techlist:
        if tech_name == 'ADX':
            # ADX动向指标
            temp_indicator = TA_ADX(high, low, close)
            # 0值填补空缺值
            temp_indicator[np.where(np.isnan(temp_indicator))[0]] = 0
            ta_indicator = np.c_[
                ta_indicator,
                temp_indicator] if len(ta_indicator) > 0 else temp_indicator

            temp_indicator = TA_ADXR(high, low, close)
            temp_indicator[np.where(np.isnan(temp_indicator))[0]] = 0
            ta_indicator = np.c_[
                ta_indicator,
                temp_indicator] if len(ta_indicator) > 0 else temp_indicator
        elif tech_name == 'MACD':
            # MACD线
            temp_indicator = TA_MACD(close)
            temp_indicator[np.where(np.isnan(temp_indicator))[0]] = 0
            ta_indicator = np.c_[
                ta_indicator,
                temp_indicator] if len(ta_indicator) > 0 else temp_indicator
        elif tech_name == 'BBANDS':
            # 布林线
            temp_indicator = TA_BBANDS(close)
            temp_indicator[np.where(np.isnan(temp_indicator))[0]] = 0
            ta_indicator = np.c_[
                ta_indicator,
                temp_indicator] if len(ta_indicator) > 0 else temp_indicator
        elif tech_name == 'CCI':
            # CCI
            temp_indicator = TA_CCI(high, low, close)
            temp_indicator[np.where(np.isnan(temp_indicator))[0]] = 0
            ta_indicator = np.c_[
                ta_indicator,
                temp_indicator] if len(ta_indicator) > 0 else temp_indicator
        elif tech_name == 'EMA':
            # 加权移动平均线
            temp_indicator = talib.EMA(close, 12).values
            temp_indicator[np.where(np.isnan(temp_indicator))[0]] = 0
            ta_indicator = np.c_[
                ta_indicator,
                temp_indicator] if len(ta_indicator) > 0 else temp_indicator

            temp_indicator = talib.EMA(close, 29).values
            temp_indicator[np.where(np.isnan(temp_indicator))[0]] = 0
            ta_indicator = np.c_[ta_indicator, temp_indicator]

            temp_indicator = talib.EMA(close, 5).values
            temp_indicator[np.where(np.isnan(temp_indicator))[0]] = 0
            ta_indicator = np.c_[ta_indicator, temp_indicator]
        elif tech_name == 'MA':
            # 移动平均线
            temp_indicator = talib.MA(close, 12).values
            temp_indicator[np.where(np.isnan(temp_indicator))[0]] = 0
            ta_indicator = np.c_[
                ta_indicator,
                temp_indicator] if len(ta_indicator) > 0 else temp_indicator

            temp_indicator = talib.MA(close, 29).values
            temp_indicator[np.where(np.isnan(temp_indicator))[0]] = 0
            ta_indicator = np.c_[ta_indicator, temp_indicator]

            temp_indicator = talib.MA(close, 5).values
            temp_indicator[np.where(np.isnan(temp_indicator))[0]] = 0
            ta_indicator = np.c_[ta_indicator, temp_indicator]
    return ta_indicator


def TA_MACD(prices: np.ndarray,
            fastperiod: int = 12,
            slowperiod: int = 26,
            signalperiod: int = 9) -> np.ndarray:

    macd, signal, hist = talib.MACD(prices,
                                    fastperiod=fastperiod,
                                    slowperiod=slowperiod,
                                    signalperiod=signalperiod)
    hist = (macd - signal) * 2
    delta = np.r_[np.nan, np.diff(hist)]
    return np.c_[macd, signal, hist, delta]


def TA_RSI(prices: np.ndarray, timeperiod: int = 12) -> np.ndarray:

    rsi = talib.RSI(prices, timeperiod=timeperiod)
    delta = np.r_[np.nan, np.diff(rsi)]
    return np.c_[rsi, delta]


def TA_BBANDS(prices: np.ndarray,
              timeperiod: int = 5,
              nbdevup: int = 2,
              nbdevdn: int = 2,
              matype: int = 0) -> np.ndarray:

    up, middle, low = talib.BBANDS(prices, timeperiod, nbdevup, nbdevdn,
                                   matype)
    ch = (up - low) / middle
    delta = np.r_[np.nan, np.diff(ch)]
    return np.c_[up, middle, low, ch, delta]


def TA_KDJ(high: np.ndarray,
           low: np.ndarray,
           close: np.ndarray,
           fastk_period: int = 9,
           slowk_matype: int = 0,
           slowk_period: int = 3,
           slowd_period: int = 3) -> np.ndarray:

    K, D = talib.STOCH(high,
                       low,
                       close,
                       fastk_period=fastk_period,
                       slowk_matype=slowk_matype,
                       slowk_period=slowk_period,
                       slowd_period=slowd_period)
    J = 3 * K - 2 * D
    delta = np.r_[np.nan, np.diff(J)]
    return np.c_[K, D, J, delta]


def TA_ADX(high: np.ndarray,
           low: np.ndarray,
           close: np.ndarray,
           timeperiod: int = 14) -> np.ndarray:

    real = talib.ADX(high, low, close, timeperiod=timeperiod)
    return np.c_[real]


def TA_ADXR(high: np.ndarray,
            low: np.ndarray,
            close: np.ndarray,
            timeperiod: int = 14) -> np.ndarray:

    real = talib.ADXR(high, low, close, timeperiod=timeperiod)
    return np.c_[real]


def TA_CCI(high: np.ndarray,
           low: np.ndarray,
           close: np.ndarray,
           timeperiod: int = 14) -> np.ndarray:

    real = talib.CCI(high, low, close, timeperiod=timeperiod)
    delta = np.r_[np.nan, np.diff(real)]
    return np.c_[real, delta]


def TA_KAMA(prices: np.ndarray, timeperiod: int = 30):

    real = talib.KAMA(prices, timeperiod=timeperiod)
    return np.c_[real]


def TA_HMA(prices: np.ndarray, timeperiod: int = 12):

    hma = talib.WMA(
        2 * talib.WMA(prices, int(timeperiod / 2)) -
        talib.WMA(prices, timeperiod), int(np.sqrt(timeperiod)))
    return hma
