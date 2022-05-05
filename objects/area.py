from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Mapping

from objects.order import Order


# random.seed(1234)
# np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)
# torch.backends.cudnn.deterministic = True

@dataclass
class Schedule:
    vehicle_id: int
    arrival_time: datetime


class Neighbor:
    def __init__(self, area_id: int, distance: float):
        self.__area_id = area_id
        self.__distance = distance

    @property
    def area_id(self) -> int:
        return self.__area_id

    @property
    def distance(self) -> float:
        return self.__distance

    def __eq__(self, other):
        if not isinstance(other, Neighbor):
            return NotImplemented
        return self.distance == other.distance    

    def __lt__(self, other):
        if not isinstance(other, Neighbor):
            return NotImplemented
        if (self.distance == 0.0) and (other.distance==0.0):
            return self.area_id < self.area_id
        else:
            return self.distance < other.distance

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

class Area:
    def __init__(
        self,
        id: int,
    ):
        self.id = id
        self.__centroid_id = None
        self._node_ids: List[int] = []
        self._neighbors: List[Neighbor] = []
        self.rebalance_number = 0
        self._idle_vehicle_ids: List[int] = []
        self._vehicles_arrive_schedule: List[Schedule] = []
        self.orders: List[Order] = []
        self.per_match_idle_vehicles = 0
        self.per_rebalance_idle_vehicles = 0
        self.later_rebalance_idle_vehicles = 0
        self.rebalance_frequency = 0
        self.dispatch_number = 0
        self.per_dispatch_idle_vehicles = 0
        self.later_dispatch_idle_vehicles = 0

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def set_node_id(self, node_id: int) -> None:
        self._node_ids.append(node_id)

    def get_node_ids(self) -> List[int]:
        return [node_id for node_id in self._node_ids]

    def set_neighbor(self, area_id: int, distance=0.0) -> None:
        self._neighbors.append(Neighbor(area_id=area_id, distance=distance))
        self._neighbors.sort()

    @property
    def area_size(self) -> int:
        return len(self._node_ids)

    @property
    def num_neighbors(self) -> int:
        return len(self._neighbors)

    def get_neighbor_ids(self) -> List[int]:
        return [neighbor.area_id for neighbor in self._neighbors]

    def set_arrival_schedule(self, vehicle_id: int, arrival_time) -> None:
        self._vehicles_arrive_schedule.append(Schedule(vehicle_id, arrival_time))

    def get_arrival_schedules(self) -> List[Schedule]:
        return [schedule for schedule in self._vehicles_arrive_schedule]

    def reset_schedule(self) -> None:
        self._vehicles_arrive_schedule.clear()

    def register_vehicle_id_as_idle_status(self, vehicle_id: int) -> None:
        self._idle_vehicle_ids.append(vehicle_id)

    def unregister_idle_vehicle_id(self, vehicle_id: int) -> None:
        self._idle_vehicle_ids.remove(vehicle_id)

    def get_idle_vehicle_ids(self) -> List[int]:
        return self._idle_vehicle_ids

    @property
    def num_idle_vehicles(self) -> int:
        return len(self._idle_vehicle_ids)

    def remove_schedule(self, schedule: Schedule) -> None:
        self._vehicles_arrive_schedule.remove(schedule)

    @property
    def centroid(self) -> int:
        return self.__centroid_id

    def set_centroid(self, centroid_id: int) -> None:
        self.__centroid_id = centroid_id

    @abstractmethod
    def example(self):
        raise NotImplementedError


class Cluster(Area):
    def reset(self):
        self.rebalance_number = 0
        self._idle_vehicle_ids.clear()
        self._vehicles_arrive_schedule.clear()
        self.orders.clear()
        self.per_rebalance_idle_vehicles = 0
        self.per_match_idle_vehicles = 0

    def example(self):
        print("Order Example output")
        print("ID:", self.id)
        print("Nodes:", self._node_ids)
        print("Neighbor:", self._node_ids)
        print("RebalanceNumber:", self.rebalance_number)
        print("IdleVehicles:", self._idle_vehicle_ids)
        print("VehiclesArrivetime:", self._vehicles_arrive_schedule)
        print("Orders:", self.orders)


class Grid(Area):
    def reset(self):
        self.rebalance_number = 0
        self._idle_vehicle_ids.clear()
        self._vehicles_arrive_schedule.clear()
        self.orders.clear()
        self.per_rebalance_idle_vehicles = 0
        self.per_match_idle_vehicles = 0

    def example(self):
        print("ID:", self.id)
        print("Nodes:", self._node_ids)
        print("Neighbor:[", end=" ")
        for i in self._neighbors:
            print(i.area_id, end=" ")
        print("]")
        print("RebalanceNumber:", self.rebalance_number)
        print("IdleVehicles:", self._idle_vehicle_ids)
        print("VehiclesArrivetime:", self._vehicles_arrive_schedule)
        print("Orders:", self.orders)
        print()


class AreaManager:
    def __init__(self):
        self.__area_list: List[Area] = None
        self.__area_dict: Dict[int, Area] = None
        self.__node_area_dict: Dict[int, int] = {}  # key = node_id, value = area_id

    def register_node_area_map(self, node_id: int, area_id: int) -> None:
        self.__node_area_dict.update({node_id: area_id})

    def node_id_to_area(self, node_id) -> Area:
        return self.__area_dict[self.__node_area_dict[node_id]]

    def set_area_list(self, area_list: List[Area]) -> None:
        self.__area_list = area_list
        self.__area_dict = {area.id: area for area in self.__area_list}

    def get_area_list(self) -> List[Area]:
        return self.__area_list

    def get_area_by_area_id(self, area_id: int) -> Area:
        return self.__area_dict[area_id]

    def get_area_list_copy(self) -> List[Area]:
        return [deepcopy(area) for area in self.__area_list]

    def reset_areas(self) -> None:
        for area in self.__area_list:
            area.reset()

    @property
    def num_areas(self) -> int:
        return len(self.__area_list)
