from dataclasses import dataclass
from copy import deepcopy
import random
from typing import List

import numpy as np
import torch

from config import Config
from domain import DispatchMode
from domain.demand_prediction_mode import DemandPredictionMode
from models import DQN
from modules.state import FeatureManager
from objects import Area, Vehicle
from objects.area import AreaManager
from objects.vehicle import VehicleManager


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.deterministic = True

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
    def __init__(self, config: Config, is_train):
        self.model = DQN(k=config.K, num_actions=config.K+1)
        if is_train:
            self.feature_manager = FeatureManager(k=config.K, mode=DemandPredictionMode.TRAIN)
        else:
            self.feature_manager = FeatureManager(k=config.K, mode=DemandPredictionMode.TEST)
        self.is_train = is_train
        self.num_neighbors = config.K

    def dispatch(self, area_manager: AreaManager, vehicle: Vehicle, prediction, supply_array, next_timeslice_datetime, feature_manager: FeatureManager = None, episode: int = 0, is_train: bool = False) -> DispatchOrder:
        current_area = area_manager.get_area_by_area_id(vehicle.location_area_id)
        candidate_area_id = ([current_area.id] + current_area.get_neighbor_ids()[:self.num_neighbors])
        # supply_array = np.array([area.num_idle_vehicles for area in area_manager.get_area_list()])
        state_list = self.feature_manager.calc_state(
            area=current_area,
            demand_array=prediction,
            supply_array=supply_array,
            next_timeslice_datetime=next_timeslice_datetime,
        )
        state_array = torch.FloatTensor(state_list)
        action = self.model.get_action(state_array, episode=episode, candidate_area_ids=candidate_area_id, is_train=is_train)
        next_area_id = candidate_area_id[action]
        next_node_id = area_manager.get_area_by_area_id(next_area_id).centroid

        if is_train:
            feature_manager.register_state(vehicle_id=vehicle.id, state_array=state_list)
            feature_manager.register_action(vehicle_id=vehicle.id, action=action)
            feature_manager.register_from_area_id(vehicle_id=vehicle.id, from_area_id=current_area.id)
            feature_manager.register_to_area_id(vehicle_id=vehicle.id, to_area_id=next_area_id)
        
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

    def __call__(self, area_manager: AreaManager, vehicle_manager: VehicleManager, prediction: np.ndarray, next_timeslice_datetime, feature_manager=None, episode: int = 0) -> List[DispatchOrder]:
        dispatch_order_list: List[DispatchOrder] = []
        supply_array = np.array([area.num_idle_vehicles for area in area_manager.get_area_list()])
        idle_vehicle_ids = []
        for area in area_manager.get_area_list():
            idle_vehicle_ids += area.get_idle_vehicle_ids()
        idle_vehicles = [vehicle_manager.get_vehicle_by_vehicle_id(i) for i in idle_vehicle_ids]
        vehicle_selector = VehicleSelector(idle_vehicles)
        for vehicle in vehicle_selector:
            dispatch_order = self.dispatch(
                area_manager=area_manager,
                vehicle=vehicle,
                episode=episode,
                supply_array=supply_array,
                prediction=prediction,
                next_timeslice_datetime=next_timeslice_datetime,
                feature_manager=feature_manager,
                is_train=self.is_train,
            )
            dispatch_order_list.append(dispatch_order)
            supply_array[dispatch_order.from_area_id] -= 1
            supply_array[dispatch_order.to_area_id] += 1
            if supply_array[dispatch_order.from_area_id] < 0:
                breakpoint()
        return dispatch_order_list


class VehicleSelector:
    def __init__(self, vehicles):
        random.shuffle(vehicles)
        self.vehicles = vehicles
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == len(self.vehicles):
            raise StopIteration()
        else:
            i = self.i
            self.i += 1
            return self.vehicles[i]
        

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
