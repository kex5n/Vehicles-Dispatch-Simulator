import numpy as np


class RewardCalculator:
    def __init__(self):
        self.omega_array = None

    def load(self, supply_array: np.ndarray, demand_array: np.ndarray) -> None:
        demand_array += 1e-3
        supply_array += 1e-3
        self.omega_array = supply_array / demand_array

    def reset(self) -> None:
        self.omega_array = None

    # @property
    # def omega_array(self) -> np.ndarray:
    #     return self.omega_array      

    @staticmethod
    def __reward(start_omega: float, distination_omega: float, is_stay: bool) -> float:
        if (0 <= start_omega) and (start_omega <= 1) and is_stay:
            return 5
        elif (0 <= start_omega) and (start_omega <= 1) and not is_stay:
            return -5
        elif (start_omega > 1) and (0 <= distination_omega) and (distination_omega <= 1):
            return 10
        elif (start_omega > 1) and (distination_omega > 1) and is_stay:
            return 0
        elif (start_omega > 1) and (distination_omega > 1) and not is_stay:
            return -1
        else:
            raise NotImplementedError
            

    def calc_reward(
        self,
        start_area_id: int,
        destination_area_id: int,
    ) -> float:
        is_stay = start_area_id == destination_area_id
        start_omega = self.omega_array[start_area_id]
        distination_omega = self.omega_array[destination_area_id]
        try:
            return RewardCalculator.__reward(
                start_omega=start_omega,
                distination_omega=distination_omega,
                is_stay=is_stay,
            )
        except:
            breakpoint()
