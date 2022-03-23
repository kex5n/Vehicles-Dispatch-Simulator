import numpy as np

from objects import Area


class StateCalculator:
    def __init__(self, supply_array: np.ndarray, demand_array: np.ndarray, k: int):
        self.k = k
        self.supply_array = supply_array
        self.demand_array = demand_array

    def calc_state(self, area: Area) -> np.ndarray:
        state_array = np.zeros(self.k*2+1)
        state_array[0] = area.id
        for i, neighbor in enumerate(area.neighbor):
            state_array[i+1] = self.supply_array[neighbor.id]
            state_array[i+self.k+1] = self.demand_array[neighbor.id]
            if i == self.k-1:
                break
        return state_array
