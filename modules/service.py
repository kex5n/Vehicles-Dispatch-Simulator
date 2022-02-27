from typing import List

import numpy as np

from objects import Order, Vehicle


class Service:
    def __init__(self, cost_map: np.ndarray):
        self.__cost_map: np.ndarray = cost_map

    def get_node_index()

    def calc_road_cost(self, start_node_index: int, end_node_index: int) -> int:
        return int(self.__cost_map[start_node_index][end_node_index])

    def find_nearest_vehicle_by_order(self, order: Order, vehicle_list: List[Vehicle]):
        tmp_min = None
        for vehicle in vehicle_list:
            tmp_road_cost = self.calc_road_cost(
                start_node_index=self.node_manager.get_node_index(
                    vehicle.location_node_id
                ),
                end_node_index=self.node_manager.get_node_index(
                    self.now_order.pick_up_node_id
                ),
            )
            if tmp_min is None:
                tmp_min = (vehicle, tmp_road_cost, order_occurred_area)
            elif tmp_road_cost < tmp_min[1]:
                tmp_min = (vehicle, tmp_road_cost, order_occurred_area)