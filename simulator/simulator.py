from collections import namedtuple
import datetime
import os
import random
from pathlib import Path
from typing import List, Mapping, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from domain import (
    AreaMode,
    ArriveInfo,
    DemandPredictionMode,
    DispatchMode,
    LocalRegionBound,
    area_mode,
)
from modules import DispatchModuleInterface, RandomDispatch, StaticsService
from modules.demand_predict import DemandPredictorInterface, MockDemandPredictor
from objects import Area, Cluster, Grid, NodeManager, Node, Order, Vehicle
from objects.order import OrderManager
from preprocessing.readfiles import read_all_files, read_map, read_order
from util import haversine

###########################################################################

DATA_PATH = "./data/Order/modified"
TRAIN = "train"
TEST = "test"

base_data_path = Path(DATA_PATH)


class Simulation(object):
    """
    This simulator is used to simulate urban vehicle traffic.The system divides the day into several time slots.
    System information is updated at the beginning of each time slot to update vehicle arrivals and order completion.
    Then the system generates the order that needs to be started within the current time slot, and then finds the optimal
    idle vehicle to match the order. If the match fails or the recent vehicles have timed out, the order is marked as Reject.
    If it is successful, the vehicle service order is arranged. The shortest path in the road network first reaches the
    place where the order occurred, and then arrives at the order destination, and repeats matching the order until all
    the orders in the current time slot have been completed. Then the system generates orders that occur within the current
    time slot, finds the nearest idle vehicle to match the order, and if there is no idle vehicle or the nearest idle vehicle
    reaches the current position of the order and exceeds the limit time, the match fails, and if the match is successful, the
    selected vehicle service is arranged Order. After the match is successful, the vehicle's idle record in the current cluster
    is deleted, and the time to be reached is added to the cluster where the order destination is located. The vehicle must
    first arrive at the place where the order occurred, pick up the passengers, and then complete the order at the order destination.
    Repeat the matching order until a match All orders in this phase are completed.
    At the end of the matching phase, you can useyour own matching method to dispatch idle vehicles in each cluster to other
    clusters that require more vehicles to meet future order requirements.
    """

    def __init__(
        self,
        area_mode: AreaMode,
        demand_prediction_mode: DemandPredictionMode,
        dispatch_mode: DispatchMode,
        vehicles_number: int,
        time_periods: np.timedelta64,
        local_region_bound: LocalRegionBound,
        side_length_meter: int,
        vehicles_server_meter: int,
        neighbor_can_server: bool,
        minutes: int,
        pick_up_time_window: np.float64,
    ):

        # Component
        self.dispatch_module: DispatchModuleInterface = self.__load_dispatch_component(
            dispatch_mode=dispatch_mode
        )
        self.demand_predictor_module: DemandPredictorInterface = self.__load_demand_prediction(
            demand_prediction_mode=demand_prediction_mode,
            area_mode=area_mode,
        )
        self.node_manager: NodeManager = None
        self.order_manager: OrderManager = None

        # Statistical variables
        self.static_service: StaticsService = StaticsService()

        # Data variable
        self.areas: List[Area] = None
        self.orders: List[Order] = None
        self.vehicles: List[Vehicle] = None
        self.__cost_map: np.ndarray = None
        self.node_id_to_area: Mapping[int, Area] = {}
        self.transition_temp_prool: List = []
        self.minutes: int = minutes
        self.pick_up_time_window: np.float64 = pick_up_time_window
        self.local_region_bound: LocalRegionBound = local_region_bound

        # Weather data
        # TODO: MUST CHANGE
        # ------------------------------------------
        # fmt: off
        self.weather_type = np.array([2,1,1,1,1,0,1,2,1,1,3,3,3,3,3,
                                     3,3,0,0,0,2,1,1,1,1,0,1,0,1,1,
                                     1,3,1,1,0,2,2,1,0,0,2,3,2,2,2,
                                     1,2,2,2,1,0,0,2,2,2,1,2,1,1,1])
        self.minimum_temperature = np.array([12,12,11,12,14,12,9,8,7,8,9,7,9,10,11,
                                            12,13,13,11,11,11,6,5,5,4,4,6,6,5,6])
        self.maximum_temperature = np.array([17,19,19,20,20,19,13,12,13,15,16,18,18,19,19,
                                            18,20,21,19,20,19,12,9,9,10,13,12,12,13,15])
        self.wind_direction = np.array([1,2,0,2,7,6,3,2,3,7,1,0,7,1,7,
                                       0,0,7,0,7,7,7,0,7,5,7,6,6,7,7])
        self.wind_power = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                   1,1,1,1,1,1,2,1,1,1,1,1,1,1,1])
        # fmt: on
        self.weather_type = self.__normalization_1d(self.weather_type)
        self.minimum_temperature = self.__normalization_1d(self.minimum_temperature)
        self.maximum_temperature = self.__normalization_1d(self.maximum_temperature)
        self.wind_direction = self.__normalization_1d(self.wind_direction)
        self.wind_power = self.__normalization_1d(self.wind_power)
        # ------------------------------------------

        # Input parameters
        self.area_mode: AreaMode = area_mode
        self.dispatch_mode: DispatchMode = dispatch_mode
        self.vehicles_number: int = vehicles_number
        self.time_periods: np.timedelta64 = time_periods
        self.side_length_meter: int = side_length_meter
        self.vehicle_service_meter: int = vehicles_server_meter
        (
            self.num_grid_width,
            self.num_grid_height,
            self.neighbor_server_deep_limit,
        ) = self.__calculate_the_scale_of_devision()
        print("----------------------------")
        print("The width of each grid", self.side_length_meter, "km")
        print("Vehicle service range", self.vehicle_service_meter, "km")
        print("Number of grids in east-west direction", self.num_grid_width)
        print("Number of grids in north-south direction", self.num_grid_height)
        print("Number of grids", self.num_areas)
        print("----------------------------")

        # Control variable
        self.neighbor_can_server = neighbor_can_server

        # Process variable
        self.real_time_in_experiment = None
        self.step = None
        self.episode = 0

        # Demand predictor variable
        self.demand_prediction_mode: DemandPredictionMode = demand_prediction_mode
        self.supply_expect = None

    @property
    def map_west_bound(self):
        return self.local_region_bound.west_bound

    @property
    def map_east_bound(self):
        return self.local_region_bound.east_bound

    @property
    def map_south_bound(self):
        return self.local_region_bound.south_bound

    @property
    def map_north_bound(self):
        return self.local_region_bound.north_bound

    @property
    def num_areas(self) -> Optional[int]:
        if (self.num_grid_width is None) or (self.num_grid_height is None):
            return None
        else:
            return self.num_grid_width * self.num_grid_height

    @property
    def total_width(self) -> float:
        return self.map_east_bound - self.map_west_bound

    @property
    def total_height(self) -> float:
        return self.map_north_bound - self.map_south_bound

    @property
    def interval_width(self) -> float:
        return self.total_width / self.num_grid_width

    @property
    def interval_height(self) -> float:
        return self.total_height / self.num_grid_height

    def create_all_instantiate(
        self,
        order_file_date: str,
    ) -> None:
        print("Read all files")
        (
            node_df,
            self.node_id_list,
            order_df,
            vehicles,
            self.__cost_map,
        ) = read_all_files(order_file_date, self.demand_prediction_mode)

        print("Create Nodes")
        self.node_manager = NodeManager(node_df)
        print("Create Orders set")
        self.order_manager = OrderManager(
            order_df=order_df, 
            pick_up_time_window=self.pick_up_time_window,
        )

        if self.area_mode == AreaMode.GRID:
            print("Create Grids")
            self.areas = self.__create_grid()
        else:
            print("Create Clusters")
            self.areas = self.__create_cluster(self.area_mode)

        for node_id in tqdm(self.node_manager.node_id_list):
            for area in self.areas:
                for node in area.nodes:
                    if node_id == node.id:
                        self.node_id_to_area[node_id] = area

        # Calculate the value of all orders in advance
        # -------------------------------
        print("Pre-calculated order value")
        for idx, each_order in enumerate(self.order_manager.get_orders()):
            cost = self.__road_cost(
                start_node_index=self.node_manager.get_node_index(
                    each_order.pick_up_node_id
                ),
                end_node_index=self.node_manager.get_node_index(
                    each_order.delivery_node_id
                ),
            )
            self.order_manager.set_order_cost(idx=idx, cost=cost)
        # -------------------------------

        # Select number of vehicles
        # -------------------------------
        vehicles = vehicles[: self.vehicles_number]
        # -------------------------------

        print("Create Vehicles set")
        self.vehicles: List[Vehicle] = [
            Vehicle(
                id=vehicle_row[0],
                location_node_id=None,
                area=None,
                orders=[],
                delivery_node_id=None,
            )
            for vehicle_row in vehicles
        ]
        self.__init_vehicles_into_area()

    def reload(self, order_file_date):
        """
        Read a new order into the simulator and
        reset some variables of the simulator
        """
        print(
            "Load order " + order_file_date + "and reset the experimental environment"
        )

        self.static_service.reset()

        self.order_manager.reset()
        self.transition_temp_prool.clear()

        self.real_time_in_experiment = None
        self.step = None

        # read orders
        # -----------------------------------------
        if self.demand_prediction_mode == DemandPredictionMode.TRAIN:
            directory = "train"
        else:
            directory = "test"
        order_df = read_order(
            input_file_path=base_data_path
            / directory
            / f"order_2016{str(order_file_date)}.csv"
        )
        self.order_manager.reload(
            order_df=order_df,
            pick_up_time_window=self.pick_up_time_window,
        )

        # Calculate the value of all orders in advance
        # -------------------------------
        for idx, each_order in enumerate(self.order_manager.get_orders()):
            cost = self.__road_cost(
                start_node_index=self.node_manager.get_node_index(
                    each_order.pick_up_node_id
                ),
                end_node_index=self.node_manager.get_node_index(
                    each_order.delivery_node_id
                ),
            )
            self.order_manager.set_order_cost(idx=idx, cost=cost)
        # -------------------------------

        # Reset the areas and Vehicles
        # -------------------------------
        for area in self.areas:
            area.reset()

        for vehicle in self.vehicles:
            vehicle.reset()

        self.__init_vehicles_into_area()
        # -------------------------------

        return

    def reset(self):
        print("Reset the experimental environment")

        self.static_service.reset()

        self.transition_temp_prool.clear()
        self.real_time_in_experiment = None
        self.step = None

        # Reset the Orders and Clusters and Vehicles
        # -------------------------------
        self.order_manager.reset()

        for area in self.areas:
            area.reset()

        for vehicle in self.vehicles:
            vehicle.reset()

        self.__init_vehicles_into_area()
        # -------------------------------
        return

    def __init_vehicles_into_area(self) -> None:
        print("Initialization Vehicles into Clusters or Grids")
        for vehicle in self.vehicles:
            random_node = random.choice(self.node_manager.get_nodes())
            vehicle.location_node_id = random_node.id
            vehicle.area = self.node_id_to_area[random_node.id]
            vehicle.area.idle_vehicles.append(vehicle)

    def __load_dispatch_component(
        self, dispatch_mode: DispatchMode
    ) -> Optional[DispatchModuleInterface]:
        if dispatch_mode == DispatchMode.RANDOM:
            dispatch_module = RandomDispatch()
            return dispatch_module
        elif dispatch_mode == DispatchMode.NOT_DISPATCH:
            return None

    def __road_cost(self, start_node_index: int, end_node_index: int) -> int:
        return int(self.__cost_map[start_node_index][end_node_index])

    def __calculate_the_scale_of_devision(self) -> Tuple[int, int, int]:

        average_longitude = (self.map_east_bound - self.map_west_bound) / 2
        average_latitude = (self.map_north_bound - self.map_south_bound) / 2

        num_grid_width = int(
            haversine(
                self.map_west_bound,
                average_latitude,
                self.map_east_bound,
                average_latitude,
            )
            / self.side_length_meter
            + 1
        )
        num_grid_height = int(
            haversine(
                average_longitude,
                self.map_south_bound,
                average_longitude,
                self.map_north_bound,
            )
            / self.side_length_meter
            + 1
        )

        neighbor_server_deep_limit = int(
            (self.vehicle_service_meter - (0.5 * self.side_length_meter))
            // self.side_length_meter
        )

        return num_grid_width, num_grid_height, neighbor_server_deep_limit


    def __create_grid(self) -> List[Area]:
        node_id_list = self.node_manager.node_id_list
        node_dict = {}
        for i in tqdm(range(len(node_id_list))):
            node_dict[
                (
                    self.node_manager.node_locations[i][0],
                    self.node_manager.node_locations[i][1],
                )
            ] = self.node_id_list.index(node_id_list[i])

        all_grid: List[Area] = [
            Grid(
                id=i,
                nodes=[],
                neighbor=[],
                rebalance_number=0,
                idle_vehicles=[],
                vehicles_arrive_time={},
                orders=[],
            )
            for i in range(self.num_areas)
        ]

        for node in self.node_manager.get_nodes():
            relatively_longitude = node.longitude - self.map_west_bound
            now_grid_width_num = int(relatively_longitude // self.interval_width)
            assert now_grid_width_num <= self.num_grid_width - 1

            relatively_latitude = node.latitude - self.map_south_bound
            now_grid_height_num = int(relatively_latitude // self.interval_height)
            assert now_grid_height_num <= self.num_grid_height - 1

            all_grid[
                self.num_grid_width * now_grid_height_num + now_grid_width_num
            ].nodes.append(node)
        # ------------------------------------------------------

        # Add neighbors to each grid
        # ------------------------------------------------------
        for grid in all_grid:

            # Bound Check
            # ----------------------------
            up_neighbor = True
            down_neighbor = True
            left_neighbor = True
            right_neighbor = True
            left_up_neighbor = True
            left_down_neighbor = True
            right_up_neighbor = True
            right_down_neighbor = True

            if grid.id >= self.num_grid_width * (self.num_grid_height - 1):
                up_neighbor = False
                left_up_neighbor = False
                right_up_neighbor = False
            if grid.id < self.num_grid_width:
                down_neighbor = False
                left_down_neighbor = False
                right_down_neighbor = False
            if grid.id % self.num_grid_width == 0:
                left_neighbor = False
                left_up_neighbor = False
                left_down_neighbor = False
            if (grid.id + 1) % self.num_grid_width == 0:
                right_neighbor = False
                right_up_neighbor = False
                right_down_neighbor = False
            # ----------------------------

            # Add all neighbors
            # ----------------------------
            if up_neighbor:
                grid.neighbor.append(all_grid[grid.id + self.num_grid_width])
            if down_neighbor:
                grid.neighbor.append(all_grid[grid.id - self.num_grid_width])
            if left_neighbor:
                grid.neighbor.append(all_grid[grid.id - 1])
            if right_neighbor:
                grid.neighbor.append(all_grid[grid.id + 1])
            if left_up_neighbor:
                grid.neighbor.append(all_grid[grid.id + self.num_grid_width - 1])
            if left_down_neighbor:
                grid.neighbor.append(all_grid[grid.id - self.num_grid_width - 1])
            if right_up_neighbor:
                grid.neighbor.append(all_grid[grid.id + self.num_grid_width + 1])
            if right_down_neighbor:
                grid.neighbor.append(all_grid[grid.id - self.num_grid_width + 1])

        return all_grid

    def __create_cluster(self, area_mode: AreaMode) -> List[Area]:
        node_location: np.ndarray = self.node_manager.node_locations
        node_id_list: np.ndarray = self.node_manager.node_id_list
        node_index_list: np.ndarray = self.node_manager.node_index_list

        N = {}
        for i in range(len(node_id_list)):
            N[(node_location[i][0], node_location[i][1])] = node_id_list[i]

        clusters = [
            Cluster(
                id=i,
                nodes=[],
                neighbor=[],
                rebalance_number=0,
                idle_vehicles=[],
                vehicles_arrive_time={},
                orders=[],
            )
            for i in range(self.num_areas)
        ]

        cluster_path = (
            "./data/"
            + f"({str(self.local_region_bound)})"
            + str(self.num_areas)
            + str(self.area_mode.value)
            + "Cluster.csv"
        )
        if os.path.exists(cluster_path):
            label_pred_df: pd.DataFrame = pd.read_csv(cluster_path)
            label_pred: np.ndarray = label_pred_df["GridID"].values
            label_pred = label_pred.flatten()
            label_pred = label_pred.astype("int64")
        else:
            raise Exception("Cluster Path not found")

        # Loading Clustering results into simulator
        print("Loading Clustering results")
        for i in range(self.num_areas):
            tmp_node_id_list = node_id_list[label_pred == i]
            tmp_node_locations = node_location[label_pred == i]
            tmp_node_index_list = node_index_list[label_pred == i]
            for node_id, node_locaton, node_index in zip(tmp_node_id_list, tmp_node_locations, tmp_node_index_list):
                clusters[i].nodes.append(
                    Node(
                        id=node_id,
                        node_index=node_index,
                        longitude=node_locaton[0],
                        latitude=node_locaton[1]
                    )
                )

        save_cluster_neighbor_path = (
            "./data/"
            + f"({str(self.local_region_bound)})"
            + str(self.num_areas)
            + str(self.area_mode)
            + "Neighbor.csv"
        )

        if not os.path.exists(save_cluster_neighbor_path):
            print("Computing Neighbor relationships between clusters")

            all_neighbor_list: List[Area] = []
            for cluster_1 in clusters:
                neighbor_list: List[Area] = []
                for cluster_2 in clusters:
                    if cluster_1 == cluster_2:
                        continue
                    else:
                        tmp_sum_cost = 0
                        for node_1 in cluster_1.nodes:
                            for node_2 in cluster_2.nodes:
                                tmp_sum_cost += self.__road_cost(
                                    start_node_index=self.node_manager.get_node_index(
                                        node_1.id
                                    ),
                                    end_node_index=self.node_manager.get_node_index(
                                        node_2.id
                                    ),
                                )
                        if (len(cluster_1.nodes) * len(cluster_2.nodes)) == 0:
                            road_network_distance = 99999
                        else:
                            road_network_distance = tmp_sum_cost / (
                                len(cluster_1.nodes) * len(cluster_2.nodes)
                            )

                    neighbor_list.append((cluster_2, road_network_distance))

                neighbor_list.sort(key=lambda X: X[1])

                all_neighbor_list.append([])
                for neighbor in neighbor_list:
                    all_neighbor_list[-1].append((neighbor[0].id, neighbor[1]))

            all_neighbor_df = pd.DataFrame(all_neighbor_list)
            all_neighbor_df.to_csv(
                save_cluster_neighbor_path, header=0, index=0
            )  # 不保存列名
            print(
                "Save the Neighbor relationship records to: "
                + save_cluster_neighbor_path
            )

        print("Load Neighbor relationship records")
        reader = pd.read_csv(save_cluster_neighbor_path, header=None, chunksize=1000)
        neighbor_list = []
        for chunk in reader:
            neighbor_list.append(chunk)
        neighbor_list_df: pd.DataFrame = pd.concat(neighbor_list)
        neighbor_list = neighbor_list_df.values

        id_to_cluster = {}
        for cluster in clusters:
            id_to_cluster[cluster.id] = cluster

        connected_threshold = 15
        for i in range(len(clusters)):
            for j in neighbor_list[i]:
                temp = eval(j)
                if len(clusters[i].neighbor) < 4:
                    clusters[i].neighbor.append(id_to_cluster[temp[0]])
                elif temp[1] < connected_threshold:
                    clusters[i].neighbor.append(id_to_cluster[temp[0]])
                else:
                    continue
        del id_to_cluster

        return clusters

    def __load_demand_prediction(
        self,
        demand_prediction_mode: DemandPredictionMode,
        area_mode: AreaMode,
    ) -> DispatchModuleInterface:
        return MockDemandPredictor(
            demand_prediction_mode=demand_prediction_mode,
            area_mode=area_mode,
        )

    def __normalization_1d(self, arr: np.ndarray) -> np.ndarray:
        arrmax = arr.max()
        arrmin = arr.min()
        arrmaxmin = arrmax - arrmin
        result = []
        for x in arr:
            x = float(x - arrmin) / arrmaxmin
            result.append(x)

        return np.array(result)

    ############################################################################

    # The main modules
    # ---------------------------------------------------------------------------
    def __demand_predict_function(
        self,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        feature: np.ndarray,
        num_areas: int
    ) -> np.ndarray:
        """
        Here you can implement your own order forecasting method
        to provide efficient and accurate help for Dispatch method
        """
        pred = self.demand_predictor_module.predict(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            feature=None,
            num_areas=num_areas
        )
        return pred

    def __supply_expect_function(self) -> None:
        """
        Calculate the number of idle Vehicles in the next time slot
        of each cluster due to the completion of the order
        """
        self.supply_expect = np.zeros(self.num_areas)
        for area in self.areas:
            for vehicle, time in list(area.vehicles_arrive_time.items()):
                # key = Vehicle ; value = Arrivetime
                if (
                    time <= self.real_time_in_experiment + self.time_periods
                    and len(vehicle.orders) > 0
                ):
                    self.supply_expect[area.id] += 1

    def __dispatch_function(self) -> None:
        """
        Here you can implement your own Dispatch method to
        move idle vehicles in each cluster to other clusters
        """
        if self.dispatch_mode == DispatchMode.RANDOM:
            for area in self.areas:
                for vehicles in area.idle_vehicles:
                    start_node_id, next_node_id, = self.dispatch_module(vehicles)
                    if start_node_id != next_node_id:
                        self.static_service.increment_dispatch_num()
                        dispatch_cost = self.__road_cost(
                            self.node_manager.get_node_index(start_node_id),
                            self.node_manager.get_node_index(next_node_id),
                        )
                        self.static_service.add_dispatch_cost(dispatch_cost)
        elif self.dispatch_mode == DispatchMode.NOT_DISPATCH:
            pass

    def __match_function(self) -> None:
        """
        Each matching module will match the orders that will occur within the current time slot.
        The matching module will find the nearest idle vehicles for each order. It can also enable
        the neighbor car search system to determine the search range according to the set search distance
        and the size of the grid. It use dfs to find the nearest idle vehicles in the area.
        """
        # Count the number of idle vehicles before matching
        for area in self.areas:
            area.per_match_idle_vehicles = len(area.idle_vehicles)

        while self.order_manager.now_order.order_time < self.real_time_in_experiment + self.time_periods:

            if not self.order_manager.has_next:
                break

            self.static_service.increment_order_num()
            order_occurred_area: Area = self.node_id_to_area[self.order_manager.now_order.pick_up_node_id]
            order_occurred_area.orders.append(self.order_manager.now_order)

            if len(order_occurred_area.idle_vehicles) or len(order_occurred_area.neighbor):
                MatchedInfo = namedtuple("MatchedInfo", ["matched_vehicle", "road_cost", "order_occurred_area"])
                matched_info = None

                if len(order_occurred_area.idle_vehicles):

                    # Find a nearest car to match the current order
                    # --------------------------------------
                    for vehicle in order_occurred_area.idle_vehicles:
                        tmp_road_cost = self.__road_cost(
                            start_node_index=self.node_manager.get_node_index(
                                vehicle.location_node_id
                            ),
                            end_node_index=self.node_manager.get_node_index(
                                self.order_manager.now_order.pick_up_node_id
                            ),
                        )
                        if matched_info is None:
                            matched_info = MatchedInfo(vehicle, tmp_road_cost, order_occurred_area)
                        elif tmp_road_cost < matched_info.road_cost:
                            matched_info = MatchedInfo(vehicle, tmp_road_cost, order_occurred_area)
                    # --------------------------------------
                # Neighbor car search system to increase search range
                elif self.neighbor_can_server and len(order_occurred_area.neighbor):
                    matched_vehicle, road_cost, order_occurred_area = self.__find_server_vehicle_function(
                        neighbor_server_deep_limit=self.neighbor_server_deep_limit,
                        visit_list={},
                        area=order_occurred_area,
                        tmp_min=None,
                        deep=0,
                    )
                    matched_info = MatchedInfo(matched_vehicle, road_cost, order_occurred_area)
                
                # When all Neighbor Cluster without any idle Vehicles
                if matched_info is None or matched_info.road_cost > self.pick_up_time_window:
                    self.static_service.increment_reject_num()
                    self.order_manager.now_order.set_arrive_info(ArriveInfo.REJECT)
                # Successfully matched a vehicle
                else:
                    matched_vehicle: Vehicle = matched_info.matched_vehicle
                    road_cost: int = matched_info.road_cost
                    order_occurred_area: Area = matched_info.order_occurred_area
                    self.order_manager.now_order.pick_up_wait_time = road_cost
                    matched_vehicle.orders.append(self.order_manager.now_order)

                    self.static_service.add_wait_time(
                        self.__road_cost(
                            start_node_index=self.node_manager.get_node_index(
                                matched_vehicle.location_node_id
                            ),
                            end_node_index=self.node_manager.get_node_index(
                                self.order_manager.now_order.pick_up_node_id
                            ),
                        )
                    )

                    schedule_cost = self.__road_cost(
                        start_node_index=self.node_manager.get_node_index(
                            matched_vehicle.location_node_id
                        ),
                        end_node_index=self.node_manager.get_node_index(
                            self.order_manager.now_order.pick_up_node_id
                        ),
                    ) + self.__road_cost(
                        start_node_index=self.node_manager.get_node_index(
                            self.order_manager.now_order.pick_up_node_id
                        ),
                        end_node_index=self.node_manager.get_node_index(
                            self.order_manager.now_order.delivery_node_id
                        ),
                    )

                    # Add a destination to the current vehicle
                    matched_vehicle.delivery_node_id = self.order_manager.now_order.delivery_node_id

                    # Delivery Cluster {Vehicle:ArriveTime}
                    self.areas[
                        self.node_id_to_area[self.order_manager.now_order.delivery_node_id].id
                    ].vehicles_arrive_time[
                        matched_vehicle
                    ] = self.real_time_in_experiment + np.timedelta64(
                        schedule_cost * self.minutes
                    )

                    # delete now Cluster's recode about now Vehicle
                    order_occurred_area.idle_vehicles.remove(matched_vehicle)
                    self.order_manager.now_order.set_arrive_info(ArriveInfo.SUCCESS)
            else:
                # None available idle Vehicles
                self.static_service.increment_reject_num()
                self.order_manager.now_order.set_arrive_info(ArriveInfo.REJECT)

            # The current order has been processed and start processing the next order
            # ------------------------------
            # breakpoint()
            self.order_manager.increment()

    def __find_server_vehicle_function(
        self, neighbor_server_deep_limit, visit_list, area: Area, tmp_min, deep
    ):
        """
        Use dfs visit neighbors and find nearest idle Vehicle
        """
        if deep > neighbor_server_deep_limit or area.id in visit_list:
            return tmp_min

        visit_list[area.id] = True
        for vehicle in area.idle_vehicles:
            tmp_road_cost = self.__road_cost(
                start_node_index=self.node_manager.get_node_index(
                    vehicle.location_node_id
                ),
                end_node_index=self.node_manager.get_node_index(
                    self.order_manager.now_order.pick_up_node_id
                ),
            )
            if tmp_min == None:
                tmp_min = (vehicle, tmp_road_cost, area)
            elif tmp_road_cost < tmp_min[1]:
                tmp_min = (vehicle, tmp_road_cost, area)

        if self.neighbor_can_server:
            for j in area.neighbor:
                tmp_min = self.__find_server_vehicle_function(
                    neighbor_server_deep_limit,
                    visit_list,
                    j,
                    tmp_min,
                    deep + 1,
                )
        return tmp_min

    def __reward_function(self) -> None:
        """
        When apply Dispatch with Reinforcement learning
        you need to implement your reward function here
        """
        return

    def __update_function(self) -> None:
        """
        Each time slot update Function will update each cluster
        in the simulator, processing orders and vehicles
        """
        for area in self.areas:
            # Records array of orders cleared for the last time slot
            area.orders.clear()
            for vehicle, time in list(area.vehicles_arrive_time.items()):
                if time <= self.real_time_in_experiment:
                    # update Vehicle info
                    vehicle.arrive_vehicle_update(area)
                    # update Cluster record
                    area.arrive_cluster_update(vehicle)

    def __get_next_state_function(self) -> None:
        """
        When apply Dispatch with Reinforcement learning
        you need to implement your next State function here
        """
        return

    def __learning_function(self) -> None:
        return

    def __call__(self) -> None:
        self.real_time_in_experiment = self.order_manager.farst_order_start_time - self.time_periods
        end_time: datetime.datetime = self.order_manager.last_order_start_time + 3 * self.time_periods
        self.step = 0

        __episode_start_time = datetime.datetime.now()
        print("Start experiment")
        print("----------------------------")
        while self.real_time_in_experiment <= end_time:

            __step_update_start_time = datetime.datetime.now()
            self.__update_function()
            self.static_service.add_update_time(
                datetime.datetime.now() - __step_update_start_time
            )

            __step_match_start_time = datetime.datetime.now()
            self.__match_function()
            self.static_service.add_match_time(
                datetime.datetime.now() - __step_match_start_time
            )

            __step_reward_start_time = datetime.datetime.now()
            self.__reward_function()
            self.static_service.add_reward_time(
                datetime.datetime.now() - __step_reward_start_time
            )

            __step_next_state_start_time = datetime.datetime.now()
            self.__get_next_state_function()
            self.static_service.add_next_state_time(
                datetime.datetime.now() - __step_next_state_start_time
            )
            for area in self.areas:
                area.dispatch_number = 0

            __step_learning_start_time = datetime.datetime.now()
            self.__learning_function()
            self.static_service.add_learning_time(
                datetime.datetime.now() - __step_learning_start_time
            )

            __step_demand_predict_start_time = datetime.datetime.now()
            self.__demand_predict_function(
                start_datetime=self.real_time_in_experiment,
                end_datetime=self.real_time_in_experiment + self.time_periods,
                feature=None,
                num_areas=self.num_areas,
            )
            self.__supply_expect_function()
            self.static_service.add_demand_predict_time(
                datetime.datetime.now() - __step_demand_predict_start_time
            )

            # Count the number of idle vehicles before Dispatch
            for area in self.areas:
                area.per_dispatch_idle_vehicles = len(area.idle_vehicles)
            step_dispatch_start_time = datetime.datetime.now()
            self.__dispatch_function()
            self.static_service.add_dispatch_time(
                datetime.datetime.now() - step_dispatch_start_time
            )
            # Count the number of idle vehicles after Dispatch
            for area in self.areas:
                area.later_dispatch_idle_vehicles = len(area.idle_vehicles)

            # A time slot is processed
            self.step += 1
            self.real_time_in_experiment += self.time_periods
        # ------------------------------------------------
        __episode_end_time = datetime.datetime.now()

        sum_order_value = 0
        order_value_num = 0
        for order in self.order_manager.get_orders():
            if order.arrive_info == ArriveInfo.SUCCESS:
                sum_order_value += order.order_value
                order_value_num += 1

        # ------------------------------------------------
        print("Experiment over")
        print(f"Episode: {self.episode}")
        print(f"Clusting mode: {self.area_mode.value}")
        print(f"Demand Prediction mode: {self.demand_prediction_mode.value}")
        print(f"Dispatch mode: {self.dispatch_mode.value}")
        print(
            "Date: "
            + str(self.order_manager.farst_order_start_time.month)
            + "/"
            + str(self.order_manager.farst_order_start_time.day)
        )
        print(
            "Weekend or Workday: "
            + self.__workday_or_weekend(self.order_manager.farst_order_start_time.weekday())
        )
        if self.area_mode == AreaMode.GRID:
            print("Number of Grids: " + str(self.num_areas))
        else:
            print("Number of Clusters: " + str(self.num_areas))
        print("Number of Vehicles: " + str(len(self.vehicles)))
        print("Number of Orders: " + str(len(self.order_manager)))
        print("Number of Reject: " + str(self.static_service.reject_num))
        print("Number of Dispatch: " + str(self.static_service.dispatch_num))
        if (self.static_service.dispatch_num) != 0:
            print(
                "Average Dispatch Cost: "
                + str(
                    self.static_service.totally_dispatch_cost
                    / self.static_service.dispatch_num
                )
            )
        if (len(self.order_manager) - self.static_service.reject_num) != 0:
            print(
                "Average wait time: "
                + str(
                    self.static_service.totally_wait_time
                    / (len(self.order_manager) - self.static_service.reject_num)
                )
            )
        print("Totally Order value: " + str(sum_order_value))
        print("Totally Update Time : " + str(self.static_service.totally_update_time))
        print(
            "Totally NextState Time : "
            + str(self.static_service.totally_next_state_time)
        )
        print(
            "Totally Learning Time : " + str(self.static_service.totally_learning_time)
        )
        print(
            "Totally Demand Predict Time : "
            + str(self.static_service.totally_demand_predict_time)
        )
        print(
            "Totally Dispatch Time : " + str(self.static_service.totally_dispatch_time)
        )
        print(
            "Totally Simulation Time : " + str(self.static_service.totally_match_time)
        )
        print("Episode Run time : " + str(__episode_end_time - __episode_start_time))

        self.static_service.save_stats(
            date=f"2016{self.order_manager.farst_order_start_time.month}{self.order_manager.farst_order_start_time.day}"
        )


        # Visualization tools
    # -----------------------------------------------
    def randomcolor(self) -> str:
        color_arr = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        ]
        color = ""
        for i in range(6):
            color += color_arr[random.randint(0, len(color_arr) - 1)]
        return "#" + color

    def draw_all_area_internal_nodes(self) -> None:
        connection_map = (read_map("./data/Map__.csv"),)
        connection_map = connection_map[0]

        areas_color = []
        for _ in range(len(self.areas)):
            areas_color.append(self.randomcolor())

        for i in tqdm(self.node_manager.node_id_list):
            for j in range(self.node_manager.node_id_list):
                if i == j:
                    continue

                if connection_map[i][j] <= 3000:
                    LX = [
                        self.node_manager.get_node(i).longitude,
                        self.node_manager.get_node(j).longitude,
                    ]
                    LY = [
                        self.node_manager.get_node(i).latitude,
                        self.node_manager.get_node(j).latitude,
                    ]

                    if self.node_id_to_area[i] == self.node_id_to_area[j]:
                        plt.plot(
                            LX,
                            LY,
                            c=areas_color[self.node_id_to_area[i].id],
                            linewidth=0.8,
                            alpha=0.5,
                        )
                    else:
                        plt.plot(LX, LY, c="grey", linewidth=0.5, alpha=0.4)

        plt.xlim(self.map_west_bound, self.map_east_bound)
        plt.ylim(self.map_south_bound, self.map_north_bound)
        plt.title(self.area_mode)
        plt.show()

    def draw_all_nodes(self) -> None:
        connection_map = (read_map("./data/Map__.csv"),)
        connection_map = connection_map[0]

        areas_color = []
        for _ in range(len(self.areas)):
            areas_color.append(self.randomcolor())

        for i in self.node_id_list:
            for j in self.node_id_list:
                if i == j:
                    continue

                if connection_map[i][j] <= 3000:
                    LX = [
                        self.node_manager.get_node(i).longitude,
                        self.node_manager.get_node(j).longitude,
                    ]
                    LY = [
                        self.node_manager.get_node(i).latitude,
                        self.node_manager.get_node(j).latitude,
                    ]

                    plt.plot(
                        LX,
                        LY,
                        c=areas_color[self.node_id_to_area[i].id],
                        linewidth=0.8,
                        alpha=0.5,
                    )

        plt.xlim(self.map_west_bound, self.map_east_bound)
        plt.ylim(self.map_south_bound, self.map_north_bound)
        plt.title(self.area_mode)
        plt.show()

    def draw_one_area(self, area: Area, random=True, show=False) -> None:
        randomc = self.randomcolor()
        for node in area.nodes:
            if random == True:
                plt.scatter(node[1][0], node[1][1], s=3, c=randomc, alpha=0.5)
            else:
                plt.scatter(node[1][0], node[1][1], s=3, c="r", alpha=0.5)
        if show == True:
            plt.xlim(self.map_west_bound, self.map_east_bound)
            plt.ylim(self.map_south_bound, self.map_north_bound)
            plt.show()

    def draw_all_vehicles(self) -> None:
        for area in self.areas:
            for vehicle in area.idle_vehicles:
                res = self.node_manager.get_node(vehicle.location_node_id)
                X = res.longitude
                Y = res.latitude
                plt.scatter(X, Y, s=3, c="b", alpha=0.3)

            for vehicle in area.vehicles_arrive_time:
                res = self.node_manager.get_node(vehicle.location_node_id)
                X = res.longitude
                Y = res.latitude
                if len(vehicle.orders):
                    plt.scatter(X, Y, s=3, c="r", alpha=0.3)
                else:
                    plt.scatter(X, Y, s=3, c="g", alpha=0.3)

        plt.xlim(self.map_west_bound, self.map_east_bound)
        plt.xlabel("red = running  blue = idle  green = Dispatch")
        plt.ylim(self.map_south_bound, self.map_north_bound)
        plt.title("Vehicles Location")
        plt.show()

    def draw_vehicle_trajectory(self, vehicle: Vehicle) -> None:
        node_1 = self.node_manager.get_node(vehicle.location_node_id)
        X1, Y1 = node_1.longitude, node_1.latitude
        node_2 = self.node_manager.get_node(vehicle.delivery_node_id)
        X2, Y2 = node_2.longitude, node_2.latitude
        # start location
        plt.scatter(X1, Y1, s=3, c="black", alpha=0.3)
        # destination
        plt.scatter(X2, Y2, s=3, c="blue", alpha=0.3)
        # Vehicles Trajectory
        LX1 = [X1, X2]
        LY1 = [Y1, Y2]
        plt.plot(LY1, LX1, c="k", linewidth=0.3, alpha=0.5)
        plt.title("Vehicles Trajectory")
        plt.show()

    # -----------------------------------------------

    def __workday_or_weekend(self, day) -> str:
        if type(day) != type(0) or day < 0 or day > 6:
            raise Exception("input format error")
        elif day == 5 or day == 6:
            return "Weekend"
        else:
            return "Workday"


"""
Experiment over
Episode: 0
Clusting mode: AreaMode.GRID
Demand Prediction mode: DemandPredictionMode.TRAINING
Dispatch mode: Simulation
Date: 6/1
Weekend or Workday: Workday
Number of Grids: 1
Number of Vehicles: 6000
Number of Orders: 130
Number of Reject: 0
Number of Dispatch: 0
Average wait time: 0.0
Totally Order value: 263
Totally Update Time : 0:00:00.000383
Totally NextState Time : 0:00:00.000068
Totally Learning Time : 0:00:00.000053
Totally Demand Predict Time : 0:00:00.000438
Totally Dispatch Time : 0:00:00.000064
Totally Simulation Time : 0:00:02.311889
Episode Run time : 0:00:02.313628

(Pdb) self.areas[0].nodes[:5]
[(0, (-74.0151098, 40.706165)), (1, (-74.015079, 40.7062557)), (2, (-74.0150681, 40.7062526)), (3, (-74.0154267, 40.7082794)), (4, (-74.0155846, 40.7078839))]


"""

"""
Experiment over
Episode: 0
Clusting mode: Grid
Demand Prediction mode: Train
Dispatch mode: NotDispatch
Date: 6/1
Weekend or Workday: Workday
Number of Grids: 9
Number of Vehicles: 50
Number of Orders: 130
Number of Reject: 37
Number of Dispatch: 0
Average wait time: 0.27956989247311825
Totally Order value: 183
Totally Update Time : 0:00:00.000368
Totally NextState Time : 0:00:00.000061
Totally Learning Time : 0:00:00.000068
Totally Demand Predict Time : 0:00:00.000546
Totally Dispatch Time : 0:00:00.000113
Totally Simulation Time : 0:00:00.005607
Episode Run time : 0:00:00.007737
"""
