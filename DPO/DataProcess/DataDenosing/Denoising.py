import numpy as np
import statsmodels.api as sm
import pywt

from pykalman import KalmanFilter


class DPODenoising:
    def __init__(self, method='wavelet'):
        self.method = method

    def denoise(self, X, *args):
        if self.method == 'wavelet':
            if len(args):
                return DPO_wavelet(X, args[0])
            else:
                return DPO_wavelet(X)

        elif self.method == 'kalman':
            if len(args):
                return DPO_kalmanfilter(X, args[0])
            else:
                return DPO_kalmanfilter(X)

        elif self.method == 'hp':
            return DPO_hpfilter(X)

        else:
            return X


def Kalman1D(observations, damping=1):

    observation_covariance = damping  # 观察偏差，越大越偏离真实值
    initial_value_guess = observations[0]
    transition_matrix = 1  # 预测时真实值的系数x_t-1
    transition_covariance = 0.5  # 初始预测偏差 H，越大越接近真实值
    kf = KalmanFilter(initial_state_mean=initial_value_guess,
                      initial_state_covariance=observation_covariance,
                      observation_covariance=observation_covariance,
                      transition_covariance=transition_covariance,
                      transition_matrices=transition_matrix)
    pred_state, state_cov = kf.smooth(observations)
    return pred_state


def wrcoef(X, coef_type, coeffs, wavename, level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))
    if coef_type == 'a':
        return pywt.upcoef('a', a, wavename, level=level)[:N]
    elif coef_type == 'd':
        return pywt.upcoef('d', ds[level - 1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))


def DPO_hpfilter(X: np.ndarray):

    _, t = sm.tsa.filters.hpfilter(X, lamb=1)

    return t


def DPO_kalmanfilter(X: np.ndarray, damping: float = 0.5):

    return Kalman1D(X, damping).reshape(-1)


def DPO_wavelet(X: np.ndarray,
                threshold_ratio: float = 0.5,
                plot: bool = False):

    coeffs = pywt.wavedec(X, 'haar', level=2)
    A2 = wrcoef(X, 'a', coeffs, 'haar', 2)
    D2 = wrcoef(X, 'd', coeffs, 'haar', 2)
    D1 = wrcoef(X, 'd', coeffs, 'haar', 1)

    threshold_num = threshold_ratio * D2.std()

    D2_less = pywt.threshold(D2,
                             value=threshold_num,
                             mode='less',
                             substitute=threshold_num)
    D2_less = pywt.threshold(D2_less,
                             value=-threshold_num,
                             mode='greater',
                             substitute=-threshold_num)

    threshold_num = 0.5 * D1.std()
    D1_less = pywt.threshold(D1,
                             value=threshold_num,
                             mode='less',
                             substitute=threshold_num)
    D1_less = pywt.threshold(D1_less,
                             value=-threshold_num,
                             mode='greater',
                             substitute=-threshold_num)

    A2_wt = A2 + D2_less + D1_less

    if plot:
        import matplotlib.pyplot as plt

        index = range(X.shape[0])
        fig = plt.figure()
        fig.set_size_inches((20, 16))
        ax_A2 = fig.add_axes((0, 0.72, 1, 0.2))
        ax_A2wt = fig.add_axes((0, 0.48, 1, 0.2), sharex=ax_A2)
        ax_D2 = fig.add_axes((0, 0.24, 1, 0.2), sharex=ax_A2)
        ax_D1 = fig.add_axes((0, 0, 1, 0.2), sharex=ax_A2)

        ax_A2.plot(index, X, label='ori')
        ax_A2.plot(index, A2, label='A2')
        ax_A2.set_title('cA2')
        ax_A2.legend()

        ax_A2wt.plot(index, X, label='ori')
        ax_A2wt.plot(index, A2_wt, label='A2')
        ax_A2wt.set_title('cA2wt')
        ax_A2wt.legend()

        ax_D2.plot(index, D2)
        ax_D2.set_title('cD2')

        ax_D1.plot(index, D1)
        ax_D1.set_title('cD1')

        fig.savefig('wave_test.png')

    return A2_wt
