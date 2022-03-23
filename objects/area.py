from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import List, Mapping, Optional, Tuple, Generator

from objects.node import Node
from objects.order import Order
from objects.vehicle import Vehicle


@dataclass
class Schedule:
    vehicle_id: int
    arrival_time: datetime


class Area:
    def __init__(
        self,
        id,
        nodes: List[Node],  # node_index: {longitude, latitude}
        neighbor: List["Area"],
        rebalance_number,
        orders,
    ):
        self.id = id
        self.__centroid = None
        self.nodes: List[Node] = nodes
        self.neighbor: List[Area] = neighbor
        self.rebalance_number = rebalance_number
        self._idle_vehicle_ids: List[int] = []
        self._vehicles_arrive_schedule: List[Schedule] = []
        self.orders: List[Order] = orders
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
    def centroid(self) -> Node:
        return self.__centroid

    def set_centroid(self, centroid: Node) -> None:
        self.__centroid = centroid

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
        print("Nodes:", self.nodes)
        print("Neighbor:", self.neighbor)
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
        print("Nodes:", self.nodes)
        print("Neighbor:[", end=" ")
        for i in self.neighbor:
            print(i.id, end=" ")
        print("]")
        print("RebalanceNumber:", self.rebalance_number)
        print("IdleVehicles:", self._idle_vehicle_ids)
        print("VehiclesArrivetime:", self._vehicles_arrive_schedule)
        print("Orders:", self.orders)
        print()


class AreaManager:
    def __init__(self):
        self.__area_list: List[Area] = None
        self.__area_dict: Mapping[int, Area] = None

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
