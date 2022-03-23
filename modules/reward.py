import numpy as np


class RewardCalculator:
    def __init__(self) -> 'RewardCalculator':
        self.__omega_array = None

    def load(self, supply_array: np.ndarray, demand_array: np.ndarray) -> None:
        self.__omega_array = supply_array / demand_array

    def reset(self) -> None:
        self.__omega_array = None        

    @staticmethod
    def __is_stay(action) -> bool:
        return action == 0

    @staticmethod
    def __reward(start_omega: float, distination_omega: float, action) -> float:
        if (0 <= start_omega) and (start_omega <= 1) and RewardCalculator.__is_stay(action):
            return 5
        elif (0 <= start_omega) and (start_omega <= 1) and not RewardCalculator.__is_stay(action):
            return -5
        elif (start_omega > 1) and (0 <= distination_omega) and (distination_omega <= 1):
            return 1 / distination_omega
        elif (start_omega > 1) and (distination_omega > 1) and RewardCalculator.__is_stay(action):
            return 0
        elif (start_omega > 1) and (distination_omega > 1) and not RewardCalculator.__is_stay(action):
            return -distination_omega
        else:
            raise NotImplementedError

    def calc_reward(
        self,
        start_area_array: np.ndarray,
        distination_area_array: np.ndarray,
        action_array: np.ndarray
    ) ->  np.ndarray:
        reward_list = []
        for start_area, distination_area, action in zip(start_area_array, distination_area_array, action_array):
            start_omega = self.__omega_array[start_area]
            distination_omega = self.__omega_array[distination_area]
            reward_list.append(
                RewardCalculator.__reward(
                    start_omega=start_omega,
                    distination_omega=distination_omega,
                    action=action,
                )
            )
        return np.ndarray(reward_list)
