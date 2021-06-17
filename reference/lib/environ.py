import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np
from . import data

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class State:
    def __init__(self,
                 bars_count,
                 commission_perc,
                 reset_on_close,
                 reward_on_close=True,
                 volumes=True):

        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes
        # TODO:补充技术指标、基础因子、数据增强、降噪的初始化部分
        # 添加数据处理过程的参数
        self.techindicator = False
        self.basefactor = False
        self.datadenosing = False
        self.dataaugment = False
        # END----------------------------------------------------------

    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count - 1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset
        # TODO:补充reset中的技术指标、基础因子、数据增强、降噪的部分
        self._TechIndicator = self.get_techindicator if self.techindicator else None
        self._BaseFactor = self.get_basefactor if self.basefactor else None

        # QUESTION: 增强和降噪过程是全局做还是滚动做
        if self.dataaugment:
            self._prices = self.data_denoising(self._prices)

        if self.datadenosing:
            self._prices = self.data_denoising(self._prices)

        # END----------------------------------------------------------

    @property
    def shape(self):
        # [h, l, c] * bars + position_flag + rel_profit (since open)
        if self.volumes:
            return (4 * self.bars_count + 1 + 1, )
        else:
            return (3 * self.bars_count + 1 + 1, )

    def data_denoising(self, prices):
        """[summary]
        数据降噪
        Args:
            prices ([type]): [description]

        Returns:
            [type]: [description]
        """
        # TODO:数据降噪部分

        # END----------------------------------------------------------

    def data_augment(self, prices):
        """[summary]
        数据增强
        Args:
            prices ([type]): [description]

        Returns:
            [type]: [description]
        """
        # TODO:数据增强部分

        # END----------------------------------------------------------

    @property
    def get_techindicator(self):
        """[summary]
        根据_prices中的基础开高收低量的数据，获取技术指标
        
        Returns:
            TechIndicator[array]: [description] m *（self.bars_count）的矩阵
                m为技术指标的数量
        """
        TechIndicator = np.ndarray(self.bars_count)
        # TODO:补充技术指标的计算部分

        # END----------------------------------------------------------
        return TechIndicator

    @property
    def get_basefactor(self):
        """[summary]
        根据_prices中的基础开高收低量的数据，获取基础因子数据
        
        Returns:
            BaseFactor[array]: [description] m *（self.bars_count）的矩阵
                m为基础因子的数量
        """
        BaseFactor = np.ndarray(self.bars_count)
        # TODO:补充基础因子的计算部分

        # END----------------------------------------------------------
        return BaseFactor

    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count + 1, 1):
            res[shift] = self._prices.high[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.low[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.close[self._offset + bar_idx]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[self._offset + bar_idx]
                shift += 1
            # TODO:补充基础类的endcode中技术指标和基础因子部分，补充滚动降噪和增强部分
            if self.techindicator:
                for idx in self._TechIndicator.shape[0]:
                    res[shift] = self._TechIndicator[idx,
                                                     self._offset + bar_idx]
                    shift += 1

            if self.basefactor:
                for idx in self._BaseFactor.shape[0]:
                    res[shift] = self._BaseFactor[idx, self._offset + bar_idx]
                    shift += 1
            # END----------------------------------------------------------
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() -
                          self.open_price) / self.open_price
        return res

    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()
        if action == Actions.Buy and not self.have_position:
            # 开仓
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc  # 开仓手续费成本
        elif action == Actions.Close and self.have_position:
            # 平仓
            reward -= self.commission_perc  # 平仓手续费成本
            done |= self.reset_on_close  # 观察是否有仓位变化
            if self.reward_on_close:  # 平仓盈利
                reward += 100.0 * (close - self.open_price) / self.open_price
            # 参数复原
            self.have_position = False
            self.open_price = 0.0

        # 窗口向前推进
        self._offset += 1
        prev_close = close
        close = self._cur_close()

        # 判断窗口是否走到最后
        done |= self._offset >= self._prices.close.shape[0] - 1

        # 在当前窗口有持仓，则使用下一个窗口的价格来计算头寸盈利加到reward中
        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close - prev_close) / prev_close

        return reward, done


class State1D(State):
    """
    State with shape suitable for 1D convolution
    """
    @property
    def shape(self):
        if self.volumes:
            return (6, self.bars_count)
        else:
            return (5, self.bars_count)

    def encode(self):

        res = np.zeros(shape=self.shape, dtype=np.float32)

        # QUESTION: 增加了滚动窗口的数据？
        ofs = self.bars_count - 1
        res[0] = self._prices.high[self._offset - ofs:self._offset + 1]
        res[1] = self._prices.low[self._offset - ofs:self._offset + 1]
        res[2] = self._prices.close[self._offset - ofs:self._offset + 1]

        # TODO:补充1D类中encode的技术指标和基础因子部分，补充数据增强和降噪的滚动部分
        # END----------------------------------------------------------
        if self.volumes:
            res[3] = self._prices.volume[self._offset - ofs:self._offset + 1]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0
            res[dst +
                1] = (self._cur_close() - self.open_price) / self.open_price

        return res


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 prices,
                 bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC,
                 reset_on_close=True,
                 state_1d=False,
                 random_ofs_on_reset=True,
                 reward_on_close=False,
                 volumes=False):
        """

        """
        assert isinstance(prices, dict)
        self._prices = prices
        if state_1d:
            self._state = State1D(bars_count,
                                  commission,
                                  reset_on_close,
                                  reward_on_close=reward_on_close,
                                  volumes=volumes)
        else:
            self._state = State(bars_count,
                                commission,
                                reset_on_close,
                                reward_on_close=reward_on_close,
                                volumes=volumes)

        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=self._state.shape,
                                                dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(prices.high.shape[0] -
                                           bars * 10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode()

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {
            file: data.load_relative(file)
            for file in data.price_files(data_dir)
        }
        return StocksEnv(prices, **kwargs)
