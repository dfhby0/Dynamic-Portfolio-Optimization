import numpy as np
import pandas as pd


# tools:一些滚动数据计算器
def LLV(Series, N=1):
    return pd.Series(Series).rolling(N).min()


def HHV(Series, N=1):
    return pd.Series(Series).rolling(N).max()


def DIFF(Series, N=1):
    return pd.Series(Series).diff(N)


def SUM(Series, N=1):
    return pd.Series.rolling(Series, N).sum()


def ABS(Series):
    return abs(Series)


def MAX(A, B):
    var = IF(A > B, A, B)
    return var


def MIN(A, B):
    var = IF(A < B, A, B)
    return var


def REF(Series, N=1):
    return Series.shift(N)


def IF(COND, V1, V2):
    var = np.where(COND, V1, V2)
    try:
        try:
            index = V1.index
        except (Exception):
            index = COND.index
    except (Exception):
        index = V2.index
    return pd.Series(var, index=index)


def IFAND(COND1, COND2, V1, V2):
    var = np.where(np.logical_and(COND1, COND2), V1, V2)
    return pd.Series(var, index=V1.index)


def IFOR(COND1, COND2, V1, V2):
    var = np.where(np.logical_or(COND1, COND2), V1, V2)
    return pd.Series(var, index=V1.index)


def COUNT(COND, N=1):
    return pd.Series(np.where(COND, 1, 0), index=COND.index).rolling(N).sum()


def STD(Series, N=1):
    return pd.Series.rolling(Series, N).std()


def MA(Series, N=1):
    return pd.Series.rolling(Series, N).mean()


def EMA(Series, N=1):
    return pd.Series.ewm(Series, span=N, min_periods=N - 1, adjust=True).mean()


def SMA(Series, N=1, M=1):
    ret = []
    i = 1
    length = len(Series)
    while i < length:
        if np.isnan(Series.iloc[i]):
            i += 1
        else:
            break
    preY = Series.iloc[i]
    ret.append(preY)
    while i < length:
        Y = (M * Series.iloc[i] + (N - M) * preY) / float(N)
        ret.append(Y)
        preY = Y
        i += 1
    return pd.Series(ret, index=Series.tail(len(ret)).index)


# technical indicator:基础技术指标
'''
MACD
BOLL
BBI
KDJ
RSI
'''


def MACD(DataFrame, short=12, long=26, mid=9):
    CLOSE = DataFrame['close']

    DIF = EMA(CLOSE, short) - EMA(CLOSE, long)
    DEA = EMA(DIF, mid)
    MACD = (DIF - DEA) * 2

    return pd.DataFrame({'DIF': DIF, 'DEA': DEA, 'MACD': MACD})


def BOLL(DataFrame, N=20, P=2):
    C = DataFrame['close']
    boll = MA(C, N)
    UB = boll + P * STD(C, N)
    LB = boll - P * STD(C, N)
    DICT = {'BOLL': boll, 'UB': UB, 'LB': LB}

    return pd.DataFrame(DICT)


def BBI(DataFrame, N1=3, N2=6, N3=12, N4=24):
    C = DataFrame['close']
    bbi = (MA(C, N1) + MA(C, N2) + MA(C, N3) + MA(C, N4)) / 4

    return pd.DataFrame({'BBI': bbi})


def KDJ(DataFrame, N=9, M1=3, M2=3):
    C = DataFrame['close']
    H = DataFrame['high']
    L = DataFrame['low']

    RSV = ((C - LLV(L, N)) / (HHV(H, N) - LLV(L, N)) *
           100).groupby('code').fillna(method='ffill')
    K = SMA(RSV, M1)
    D = SMA(K, M2)
    J = 3 * K - 2 * D

    return pd.DataFrame({'KDJ_K': K, 'KDJ_D': D, 'KDJ_J': J})


def RSI(DataFrame, N=12):
    CLOSE = DataFrame['close']
    LC = REF(CLOSE, 1)
    RSI = SMA(MAX(CLOSE - LC, 0), N) / SMA(ABS(CLOSE - LC), N) * 100

    return pd.DataFrame({'RSI': RSI})
