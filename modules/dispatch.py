import random
from typing import Tuple

from objects import Area, Vehicle


class DispatchModuleInterface:
    def __call__(self, vehicle: Vehicle) -> bool:
        raise NotImplementedError


class RandomDispatch(DispatchModuleInterface):
    def __call__(self, vehicle: Vehicle) -> Tuple[int, int]:
        current_area: Area = vehicle.area
        candidate_area = current_area.neighbor + [current_area]
        next_area = random.choice(candidate_area)
        next_node = random.choice(next_area.nodes)
        next_node_id = next_node.id
        start_node_id = vehicle.location_node_id
        vehicle.area = next_area
        vehicle.location_node_id = next_node.id

        return int(start_node_id), int(next_node_id)
