from typing import List

import numpy as np

from objects.order import Order


class Vehicle(object):
    def __init__(self, id: int):
        self.id = id  # This vehicle's ID
        self.__location_node_id = None  # Current vehicle's location
        self.__location_area_id = None  # Which cluster the current vehicle belongs to
        self.orders: List[Order] = []  # Orders currently on board
        self.delivery_node_id = None  # Next destination of current vehicle

    def move(self, destination_area_id: int) -> None:
        self.__location_node_id = self.delivery_node_id
        self.delivery_node_id = None
        self.__location_area_id = destination_area_id
        if len(self.orders):
            self.orders.clear()

    def deploy_to_node(self, location_node_id: int) -> None:
        self.__location_node_id = location_node_id

    def deploy_to_area(self, location_area_id: int) -> None:
        self.__location_area_id = location_area_id

    @property
    def location_node_id(self):
        return self.__location_node_id

    @property
    def location_area_id(self):
        return self.__location_area_id

    def reset(self) -> None:
        self.orders.clear()
        self.delivery_node_id = None

    def example(self) -> None:
        print("Vehicle Example output")
        print("ID:", self.id)
        print("LocationNode:", self.__location_node_id)
        print("Area:", self.__location_area_id)
        print("Orders:", self.orders)
        print("DeliveryPoint:", self.delivery_node_id)
        print()


class VehicleManager:
    def __init__(self, vehicle_array: np.array):
        self.__vehicle_list: List[Vehicle] = [
            Vehicle(id=vehicle_row[0],)
            for vehicle_row in vehicle_array
        ]
        self.__vehicle_dict = {vehicle.id: vehicle for vehicle in self.__vehicle_list}

    def get_vehicle_list(self) -> List[Vehicle]:
        return [vehicle for vehicle in self.__vehicle_list]

    def get_vehicle_by_vehicle_id(self, vehicle_id: int) -> Vehicle:
        return self.__vehicle_dict[vehicle_id]

    def reset_vehicles(self) -> None:
        for vehicle in self.__vehicle_list:
            vehicle.reset()
