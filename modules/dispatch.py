from dataclasses import dataclass
import random
from typing import Tuple, List

from domain import DispatchMode
from objects import Area, Node, Vehicle
from objects.area import AreaManager
from objects.vehicle import VehicleManager


class DispatchOrder:
    def __init__(self, vehicle_id: int, start_node_id: int, end_node_id: int):
        self.vehicle_id = vehicle_id
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id


class DispatchModuleInterface:
    def dispatch(self, area_manager: AreaManager, vehicle: Vehicle) -> DispatchOrder:
        raise NotImplementedError


class RandomDispatch(DispatchModuleInterface):
    def dispatch(self, area_manager: AreaManager, vehicle: Vehicle) -> DispatchOrder:
        current_area: Area = area_manager.get_area_by_area_id(vehicle.location_area_id)
        candidate_area = current_area.neighbor + [current_area]
        next_area = random.choice(candidate_area)
        next_node = next_area.centroid
        next_node_id = next_node.id
        start_node_id = vehicle.location_node_id

        return DispatchOrder(
            vehicle_id=vehicle.id,
            start_node_id=start_node_id,
            end_node_id=next_node_id,
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

def load_dispatch_component(dispatch_mode: DispatchMode) -> DispatchModuleInterface:
    if dispatch_mode == DispatchMode.RANDOM:
        dispatch_module = RandomDispatch()
        return dispatch_module
    elif dispatch_mode == DispatchMode.NOT_DISPATCH:
        return None
