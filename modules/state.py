from typing import Any, Dict, Tuple
import numpy as np

from domain.demand_prediction_mode import DemandPredictionMode

# random.seed(1234)
np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)
# torch.backends.cudnn.deterministic = True

class FeatureManager:
    def __init__(self, k: int, mode: DemandPredictionMode):
        self.k = k
        self.__feature_dict: Dict = {}
        self.area_scale_dict = {i: self.area_scale(i) for i in range(53)}
        self.target_scale_dict = {i: self.target_scale(i) for i in range(2001)}
        self.hour_sign_dict = {i: np.sin(i*15) for i in range(24)}
        self.hour_cos_dict = {i: np.cos(i*15) for i in range(24)}
        self.minute_sign_dict = {i: np.sign(i*60) for i in range(0,60,10)}
        self.minute_cos_dict = {i: np.cos(i*60) for i in range(0,60,10)}
        self.mode = mode

    def reset(self) -> None:
        self.__feature_dict: Dict = {}

    @staticmethod
    def area_scale(x):
        return (x - 26) / 15.29705854

    @staticmethod
    def target_scale(x):
        return (x - 5.67915983) / 8.85665129


    def calc_state(self, area, demand_array: np.ndarray, supply_array: np.ndarray, next_timeslice_datetime) -> np.ndarray:
        # 
        # ["GridID", "month", "day", "hour", "minute", "before"]
        # ss.mean_ = array([26, 6, 12, 11.5, 25, 5.67915983])
        # ss.scale_ = array([15.29705854, 1, 6.63324958, 6.92218655, 17.07825128, 8.85665129])
        #

        state_array = np.zeros(3+(self.k)*2+4)
        state_array[0] = self.area_scale_dict[area.id]

        if self.mode == DemandPredictionMode.TRAIN:
            state_array[1] = self.target_scale_dict[demand_array[area.id]]
        else:
            state_array[1] = self.target_scale(demand_array[area.id])
        state_array[2] = self.target_scale_dict[supply_array[area.id]]

        for i, neighbor_area_id in enumerate(area.get_neighbor_ids()):
            if self.mode == DemandPredictionMode.TRAIN:
                state_array[(i+1)*2+1] = self.target_scale_dict[demand_array[neighbor_area_id]]
            else:
                state_array[(i+1)*2+1] = self.target_scale(demand_array[neighbor_area_id])  # num_demand
            state_array[(i+1)*2+2] = self.target_scale_dict[supply_array[neighbor_area_id]]  # num_supply

            if i == self.k-1:
                state_array[-4] = self.hour_sign_dict[next_timeslice_datetime.hour]
                state_array[-3] = self.hour_cos_dict[next_timeslice_datetime.hour]
                state_array[-2] = self.minute_sign_dict[next_timeslice_datetime.minute]
                state_array[-1] = self.minute_cos_dict[next_timeslice_datetime.minute]
                break
        return state_array

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
