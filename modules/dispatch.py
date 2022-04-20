from dataclasses import dataclass
import random
from typing import List

import numpy as np
import torch

from config import Config
from domain import DispatchMode
from models import DQN
from modules.state import FeatureManager
from objects import Area, Vehicle
from objects.area import AreaManager
from objects.vehicle import VehicleManager


@dataclass(frozen=True)
class DispatchOrder:
    vehicle_id: int
    start_node_id: int
    end_node_id: int
    action: int
    from_area_id: int = None
    to_area_id: int = None


class DispatchModuleInterface:
    def dispatch(self, area_manager: AreaManager, vehicle: Vehicle) -> DispatchOrder:
        raise NotImplementedError

    def __call__(self, area_manager: AreaManager, vehicle_manager: VehicleManager) -> List[DispatchOrder]:
        raise NotImplementedError


class RandomDispatch(DispatchModuleInterface):
    def dispatch(self, area_manager: AreaManager, vehicle: Vehicle) -> DispatchOrder:
        current_area: Area = area_manager.get_area_by_area_id(vehicle.location_area_id)
        candidate_area_id = [current_area.id] + current_area.get_neighbor_ids()
        next_area = area_manager.get_area_by_area_id(random.choice(candidate_area_id))
        next_node_id = next_area.centroid
        start_node_id = vehicle.location_node_id

        return DispatchOrder(
            vehicle_id=vehicle.id,
            start_node_id=start_node_id,
            end_node_id=next_node_id,
            action=None,
        )

    def __call__(self, area_manager: AreaManager, vehicle_manager: VehicleManager) -> List[DispatchOrder]:
        dispatch_order_list: List[DispatchOrder] = []
        for area in area_manager.get_area_list():
            for vehicle_id in area.get_idle_vehicle_ids():
                vehicle = vehicle_manager.get_vehicle_by_vehicle_id(vehicle_id)
                dispatch_order = self.dispatch(
                    area_manager=area_manager,
                    vehicle=vehicle,
                )
                dispatch_order_list.append(dispatch_order)
        return dispatch_order_list


class DQNDispatch(DispatchModuleInterface):
    def __init__(self, config: Config, is_train=False):
        self.model = DQN(k=config.K, num_actions=9)
        self.__feature_manager = FeatureManager(k=config.K)
        self.is_train = is_train

    def dispatch(self, area_manager: AreaManager, vehicle: Vehicle, prediction, episode: int = 0, is_train: bool = False) -> DispatchOrder:
        current_area = area_manager.get_area_by_area_id(vehicle.location_area_id)
        candidate_area_id = [current_area.id] + current_area.get_neighbor_ids()
        supply_array = np.array([area.num_idle_vehicles for area in area_manager.get_area_list()])
        state_list = self.__feature_manager.calc_state(
            area=current_area,
            demand_array=prediction,
            supply_array=supply_array
        )
        state_array = torch.FloatTensor(state_list)
        action = self.model.get_action(state_array, episode=episode, candidate_area_ids=candidate_area_id, is_train=is_train)
        # if candidate_area_id[0] == 3:
        #     breakpoint()
        next_area_id = candidate_area_id[action]
        next_node_id = area_manager.get_area_by_area_id(next_area_id).centroid
        return DispatchOrder(
            vehicle_id=vehicle.id,
            start_node_id=vehicle.location_node_id,
            end_node_id=next_node_id,
            action=action,
            from_area_id=current_area.id,
            to_area_id=next_area_id
        )

    def memorize(self, state, action, next_state, reward, from_area_id, to_area_id) -> None:
        self.model.memorize(state, action, next_state, reward, from_area_id, to_area_id)

    def train(self, area_manager: AreaManager, date_info, episode=None):
        return self.model.update_q_function(area_manager=area_manager, date_info=date_info, episode=episode)

    def save(self, checkpoint_path: str) -> None:
        self.model.save_checkpoint(checkpoint_path)

    def load(self, checkpoint_path: str) -> None:
        self.model.load_checkpoint(checkpoint_path)

    def __call__(self, area_manager: AreaManager, vehicle_manager: VehicleManager, prediction: np.ndarray, episode: int = 0) -> List[DispatchOrder]:
        dispatch_order_list: List[DispatchOrder] = []
        for area in area_manager.get_area_list():
            for vehicle_id in area.get_idle_vehicle_ids():
                vehicle = vehicle_manager.get_vehicle_by_vehicle_id(vehicle_id)
                dispatch_order = self.dispatch(
                    area_manager=area_manager,
                    vehicle=vehicle,
                    episode=episode,
                    prediction=prediction,
                    is_train=self.is_train,
                )
                dispatch_order_list.append(dispatch_order)
        return dispatch_order_list
        

def load_dispatch_component(dispatch_mode: DispatchMode, config: Config, is_train=False) -> DispatchModuleInterface:
    if dispatch_mode == DispatchMode.DQN:
        dispatch_module = DQNDispatch(config=config, is_train=is_train)
        return dispatch_module
    elif dispatch_mode == DispatchMode.RANDOM:
        dispatch_module = RandomDispatch()
        return dispatch_module
    elif dispatch_mode == DispatchMode.NOT_DISPATCH:
        return None
    else:
        raise NotImplementedError
