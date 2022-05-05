from typing import Any, Dict, Tuple
import numpy as np

# random.seed(1234)
np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)
# torch.backends.cudnn.deterministic = True

class FeatureManager:
    def __init__(self, k: int):
        self.k = k
        self.__feature_dict: Dict = {}

    def reset(self) -> None:
        self.__feature_dict: Dict = {}

    def calc_state(self, area, demand_array: np.ndarray, supply_array: np.ndarray) -> np.ndarray:
        state_array = np.zeros((self.k+1)*2 +1)
        state_array[0] = area.id
        state_array[1] = demand_array[area.id]
        state_array[2] = supply_array[area.id]
        for i, neighbor_area_id in enumerate(area.get_neighbor_ids()):
            state_array[(i+1)*2+1] = demand_array[neighbor_area_id]  # num_supply
            state_array[(i+1)*2+2] = supply_array[neighbor_area_id]  # num_demand
            if i == self.k-1:
                break
        return state_array / 10

    def __create_record(self, vehicle_id: int) -> None:
        self.__feature_dict.update(
            {
                vehicle_id: {
                    "state": None,
                    "next_state": None,
                    "action": None,
                    "reward": None,
                    "from_area_id": None,
                    "to_area_id": None,
                }
            }
        )

    def register_state(self, vehicle_id: int, state_array: np.ndarray) -> None:
        if vehicle_id not in self.__feature_dict.keys():
            self.__create_record(vehicle_id)
        self.__feature_dict[vehicle_id]["state"] = state_array

    def register_next_state(self, vehicle_id: int, next_state_array: np.ndarray) -> None:
        if vehicle_id not in self.__feature_dict.keys():
            self.__create_record(vehicle_id)
        self.__feature_dict[vehicle_id]["next_state"] = next_state_array

    def register_action(self, vehicle_id: int, action: int) -> None:
        if vehicle_id not in self.__feature_dict.keys():
            self.__create_record(vehicle_id)
        self.__feature_dict[vehicle_id]["action"] = action

    def register_reward(self, vehicle_id: int, reward: float) -> None:
        if vehicle_id not in self.__feature_dict.keys():
            self.__create_record(vehicle_id)
        self.__feature_dict[vehicle_id]["reward"] = reward

    def register_from_area_id(self, vehicle_id: int, from_area_id: int) -> None:
        if vehicle_id not in self.__feature_dict.keys():
            self.__create_record(vehicle_id)
        self.__feature_dict[vehicle_id]["from_area_id"] = from_area_id

    def register_to_area_id(self, vehicle_id: int, to_area_id: int) -> None:
        if vehicle_id not in self.__feature_dict.keys():
            self.__create_record(vehicle_id)
        self.__feature_dict[vehicle_id]["to_area_id"] = to_area_id

    def get_from_area_id_by_vehicle_id(self, vehicle_id: int) -> int:
        return self.__feature_dict[vehicle_id]["from_area_id"]

    def get_to_area_id_by_vehicle_id(self, vehicle_id: int) -> int:
        return self.__feature_dict[vehicle_id]["to_area_id"]

    # for debug
    def get_action_by_vehicle_id(self, vehicle_id: int) -> int:
        return self.__feature_dict[vehicle_id]["action"]

    def get_whole_data(self) -> Tuple[int, Dict[str, Any]]:
        return [(vehicle_id, feature) for vehicle_id, feature in self.__feature_dict.items()]
