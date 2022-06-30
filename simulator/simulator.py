from collections import namedtuple
from copy import deepcopy
import datetime
import glob
import json
import os
import random
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional, Tuple

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
)
from modules import StaticsService
from modules.demand_predict import DemandPredictorInterface, load_demand_prediction_component
from modules.dispatch import DispatchOrder
from objects import (
    Area,
    AreaManager,
    Cluster,
    Grid,
    NodeManager,
    OrderManager,
    Vehicle,
    VehicleManager,
)
from objects.order import Order
from preprocessing.readfiles import read_all_files, read_order
from util import haversine


random.seed(1234)
np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)
# torch.backends.cudnn.deterministic = True

###########################################################################

DATA_PATH = "./data/medium/Order/modified"

base_data_path = Path(DATA_PATH)


class Simulator(object):
    def __init__(
        self,
        area_mode: AreaMode,
        demand_prediction_mode: DemandPredictionMode,
        dispatch_mode: DispatchMode,
        data_size: str,
        vehicles_number: int,
        time_periods: np.timedelta64,
        local_region_bound: LocalRegionBound,
        side_length_meter: int,
        vehicles_server_meter: int,
        neighbor_can_server: bool,
        minutes: int,
        pick_up_time_window: np.float64,
        is_train: bool,
        debug_: bool =False
    ):
        self.__node_manager: NodeManager = None
        self.order_manager: OrderManager = None
        self.area_manager: AreaManager = AreaManager()
        self.vehicle_manager: VehicleManager = None
        self.dispatch_order_list: List[DispatchOrder] = None

        # Statistical variables
        self.__static_service: StaticsService = StaticsService()

        # Data variable
        self.__cost_map: np.ndarray = None
        self.__transition_temp_prool: List = []
        self.__minutes: int = minutes
        self.__pick_up_time_window: np.float64 = pick_up_time_window
        self.__local_region_bound: LocalRegionBound = local_region_bound
        self.data_size = data_size
        self.is_train = is_train
        self.debug_: bool = debug_

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
        return self.__local_region_bound.west_bound

    @property
    def map_east_bound(self):
        return self.__local_region_bound.east_bound

    @property
    def map_south_bound(self):
        return self.__local_region_bound.south_bound

    @property
    def map_north_bound(self):
        return self.__local_region_bound.north_bound

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
        ) = read_all_files(order_file_date, self.demand_prediction_mode, self.data_size, debug_=self.debug_)

        print("Create Nodes")
        self.__node_manager = NodeManager(node_df)
        print("Create Orders set")
        self.order_manager = OrderManager(
            order_df=order_df,
            pick_up_time_window=self.__pick_up_time_window,
        )

        if self.area_mode == AreaMode.GRID:
            print("Create Grids")
            self.area_manager.set_area_list(self.__create_grid())
            print(f"number of grid: {self.area_manager.num_areas}")
        else:
            print("Create Clusters")
            self.area_manager.set_area_list(self.__create_cluster())
            print(f"number of cluster: {self.area_manager.num_areas}")

        for area in self.area_manager.get_area_list():
            for node_id in area.get_node_ids():
                self.area_manager.register_node_area_map(node_id=node_id, area_id=area.id)

        # Calculate the value of all orders in advance
        # -------------------------------
        print("Pre-calculated order value")
        for idx, each_order in enumerate(self.order_manager.get_orders()):
            cost = self.__road_cost(
                start_node_index=self.__node_manager.get_node_index(
                    each_order.pick_up_node_id
                ),
                end_node_index=self.__node_manager.get_node_index(
                    each_order.delivery_node_id
                ),
            )
            self.order_manager.set_order_cost(idx=idx, cost=cost)
        # -------------------------------

        # Select number of vehicles
        # -------------------------------
        vehicles = vehicles[:self.vehicles_number]
        # -------------------------------

        print("Create Vehicles set")
        self.vehicle_manager = VehicleManager(vehicle_array=vehicles)
        self.__init_vehicles_into_area(
            area_manager=self.area_manager,
            node_manager=self.__node_manager,
            vehicle_manager=self.vehicle_manager,
        )
        # breakpoint()

    def reload(self, order_file_date):
        """
        Read a new order into the simulator and
        reset some variables of the simulator
        """
        print(
            "Load order " + order_file_date + " and reset the experimental environment"
        )

        self.__static_service.reset()

        self.order_manager.reset()
        self.__transition_temp_prool.clear()

        self.real_time_in_experiment = None
        self.step = None

        # read orders
        # -----------------------------------------
        if self.demand_prediction_mode == DemandPredictionMode.TRAIN:
            directory = "train"
        else:
            directory = "test"
        # if True:
        #     directory = "dummy"
        order_df = read_order(
            input_file_path=base_data_path
            / directory
            / f"order_2016{str(order_file_date)}.csv"
        )
        self.order_manager.reload(
            order_df=order_df,
            pick_up_time_window=self.__pick_up_time_window,
        )

        # Calculate the value of all orders in advance
        # -------------------------------
        for idx, each_order in enumerate(self.order_manager.get_orders()):
            cost = self.__road_cost(
                start_node_index=self.__node_manager.get_node_index(
                    each_order.pick_up_node_id
                ),
                end_node_index=self.__node_manager.get_node_index(
                    each_order.delivery_node_id
                ),
            )
            self.order_manager.set_order_cost(idx=idx, cost=cost)
        # -------------------------------

        # Reset the areas and Vehicles
        # -------------------------------
        self.area_manager.reset_areas()
        self.vehicle_manager.reset_vehicles()

        self.__init_vehicles_into_area(
            area_manager=self.area_manager,
            node_manager=self.__node_manager,
            vehicle_manager=self.vehicle_manager,
        )
        # -------------------------------

        return

    def reset(self, order_file_date):
        print("Reset the experimental environment")
        self.__static_service.reset()

        self.__transition_temp_prool.clear()
        self.real_time_in_experiment = None
        self.step = None

        # Reset the Orders and Clusters and Vehicles
        # -------------------------------
        self.order_manager.reset()
        # read orders
        # -----------------------------------------
        if self.demand_prediction_mode == DemandPredictionMode.TRAIN:
            directory = "train"
        else:
            directory = "test"
        # if True:
        #     directory = "dummy"
        order_df = read_order(
            input_file_path=base_data_path
            / directory
            / f"order_2016{str(order_file_date)}.csv"
        )
        self.order_manager.reload(
            order_df=order_df,
            pick_up_time_window=self.__pick_up_time_window,
        )

        self.area_manager.reset_areas()
        self.vehicle_manager.reset_vehicles()

        self.__init_vehicles_into_area(
            area_manager=self.area_manager,
            node_manager=self.__node_manager,
            vehicle_manager=self.vehicle_manager,
        )
        # -------------------------------
        return

    def __init_vehicles_into_area(
        self,
        area_manager: AreaManager,
        node_manager: NodeManager,
        vehicle_manager: VehicleManager,
    ) -> None:
        candidate_areas = [area for area in area_manager.get_area_list()]
        for vehicle in vehicle_manager.get_vehicle_list():
            random_area = random.choice(candidate_areas)
            random_node_id = random.choice(random_area.get_node_ids())
            vehicle.deploy_to_node(random_node_id)            
            vehicle.deploy_to_area(random_area.id)
            random_area.register_vehicle_id_as_idle_status(vehicle.id)

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
        node_id_list = self.__node_manager.node_id_list
        node_location_list = self.__node_manager.node_locations
        df = pd.DataFrame(columns=["NodeID", "Longitude", "Latitude"])
        df["NodeID"] = node_id_list
        df["Longitude"] = node_location_list[:, 0]
        df["Latitude"] = node_location_list[:, 1]
        nn = NearestNeighbors(algorithm='ball_tree')
        nn.fit(df[["Longitude", "Latitude"]].values)

        node_dict = {}
        for i in tqdm(range(len(node_id_list))):
            node_dict[
                (
                    self.__node_manager.node_locations[i][0],
                    self.__node_manager.node_locations[i][1],
                )
            ] = self.node_id_list.index(node_id_list[i])

        all_grid: List[Area] = [Grid(id=i) for i in range(self.num_areas)]

        for i in range(self.num_grid_height):
            for j in range(self.num_grid_width):
                center_longitude = self.map_west_bound + self.interval_width * j + self.interval_width / 2
                center_latitude = self.map_south_bound + self.interval_height * i + self.interval_height / 2
                _, start_indices = nn.kneighbors(np.array([[center_longitude, center_latitude]]), n_neighbors=1)
                centroid_node = self.__node_manager.get_node(node_id_list[start_indices[0][0]])
                all_grid[self.num_grid_width*i + j].set_centroid(centroid_node.id)

        for node in self.__node_manager.get_node_list():
            relatively_longitude = node.longitude - self.map_west_bound
            now_grid_width_num = int(relatively_longitude // self.interval_width)
            assert now_grid_width_num <= self.num_grid_width - 1

            relatively_latitude = node.latitude - self.map_south_bound
            now_grid_height_num = int(relatively_latitude // self.interval_height)
            assert now_grid_height_num <= self.num_grid_height - 1

            all_grid[
                self.num_grid_width * now_grid_height_num + now_grid_width_num
            ].set_node_id(node.id)
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
                grid.set_neighbor(all_grid[grid.id + self.num_grid_width].id)
            if down_neighbor:
                grid.set_neighbor(all_grid[grid.id - self.num_grid_width].id)
            if left_neighbor:
                grid.set_neighbor(all_grid[grid.id - 1].id)
            if right_neighbor:
                grid.set_neighbor(all_grid[grid.id + 1].id)
            if left_up_neighbor:
                grid.set_neighbor(all_grid[grid.id + self.num_grid_width - 1].id)
            if left_down_neighbor:
                grid.set_neighbor(all_grid[grid.id - self.num_grid_width - 1].id)
            if right_up_neighbor:
                grid.set_neighbor(all_grid[grid.id + self.num_grid_width + 1].id)
            if right_down_neighbor:
                grid.set_neighbor(all_grid[grid.id - self.num_grid_width + 1].id)

        return all_grid

    def __create_cluster(self) -> List[Area]:
        node_id_list: np.ndarray = self.__node_manager.node_id_list
        cluster_path = glob.glob(f"./data/{self.data_size}/*{str(self.area_mode.value)}Cluster.csv")[0]
        cluster_ids = set(pd.read_csv(cluster_path)["GridID"])
        clusters = [Cluster(id=i) for i in cluster_ids]
        if os.path.exists(cluster_path):
            label_pred_df: pd.DataFrame = pd.read_csv(cluster_path)
            label_pred: np.ndarray = label_pred_df["GridID"].values
            label_pred = label_pred.flatten()
            label_pred = label_pred.astype("int64")
        else:
            raise Exception("Cluster Path not found")

        # Loading Clustering results into simulator
        print("Loading Clustering results")
        for i, cluster_id in enumerate(cluster_ids):
            tmp_node_id_list = node_id_list[label_pred == cluster_id]
            try:
                centroid_node_id = label_pred_df[(label_pred_df["GridID"]==cluster_id) & (label_pred_df["IsCentroid"]==True)]["NodeID"].values[0]
            except:
                breakpoint()
            clusters[i].set_centroid(centroid_node_id)
            for node_id in tmp_node_id_list:
                clusters[i].set_node_id(node_id)

        save_cluster_neighbor_path = (
            "./data/"
            + self.data_size + "/"
            + f"({str(self.__local_region_bound)})"
            + str(len(cluster_ids))
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
                        for node_id_1 in cluster_1.get_node_ids():
                            for node_id_2 in cluster_2.get_node_ids():
                                tmp_sum_cost += self.__road_cost(
                                    start_node_index=self.__node_manager.get_node_index(
                                        node_id_1
                                    ),
                                    end_node_index=self.__node_manager.get_node_index(
                                        node_id_2
                                    ),
                                )
                        if (cluster_1.area_size * cluster_2.area_size) == 0:
                            road_network_distance = 99999
                        else:
                            road_network_distance = tmp_sum_cost / (
                                cluster_1.area_size * cluster_2.area_size
                            )

                    neighbor_list.append((cluster_2, road_network_distance))

                neighbor_list.sort(key=lambda X: X[1])

                all_neighbor_list.append([])
                for neighbor in neighbor_list:
                    all_neighbor_list[-1].append(json.dumps({"area_id": neighbor[0].id, "distance": neighbor[1]}))

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

        connected_threshold = 15
        for i in range(len(clusters)):
            for j in neighbor_list[i]:
                neighbor_info = json.loads(j)
                if len(clusters[i].get_neighbor_ids()) < 4:
                    clusters[i].set_neighbor(area_id=neighbor_info["area_id"], distance=neighbor_info["distance"])
                elif neighbor_info["distance"] < connected_threshold:
                    clusters[i].set_neighbor(neighbor_info["area_id"], distance=neighbor_info["distance"])
                else:
                    continue

        return clusters

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

    def __supply_expect_function(self) -> None:
        """
        Calculate the number of idle Vehicles in the next time slot
        of each cluster due to the completion of the order
        """
        self.supply_expect = np.zeros(self.num_areas)
        for area in self.area_manager.get_area_list():
            for schedule in list(area.get_arrival_schedules()):
                # key = Vehicle ; value = Arrivetime
                vehicle = self.vehicle_manager.get_vehicle_by_vehicle_id(schedule.vehicle_id)
                if (
                    schedule.arrival_time <= self.real_time_in_experiment + self.time_periods
                    and len(vehicle.orders) > 0
                ):
                    self.supply_expect[area.id] += 1

    def set_dispatch_orders(self, dispatch_order_list: List[DispatchOrder]) -> None:
        self.dispatch_order_list = dispatch_order_list.copy()
        self.vehicle_manager.reset_is_dispatched()
        for dispatch_order in dispatch_order_list:
            self.vehicle_manager.get_vehicle_by_vehicle_id(dispatch_order.vehicle_id).set_is_dispatched(True)

    @property
    def area_manager_copy(self):
        return deepcopy(self.area_manager)

    @property
    def vehicle_manager_copy(self):
        return deepcopy(self.vehicle_manager)

    @property
    def order_manager_copy(self):
        return deepcopy(self.order_manager)

    def match(self) -> None:
        self.__match_function()

    def __match_function(self) -> None:
        """
        Each matching module will match the orders that will occur within the current time slot.
        The matching module will find the nearest idle vehicles for each order. It can also enable
        the neighbor car search system to determine the search range according to the set search distance
        and the size of the grid. It use dfs to find the nearest idle vehicles in the area.
        """
        match_count = {i:0 for i in range(self.area_manager.num_areas)}

        for area in self.area_manager.get_area_list():
            area.per_match_idle_vehicles = area.num_idle_vehicles

        end_time = self.real_time_in_experiment + self.time_periods
        while self.order_manager.now_order.order_time < end_time:
            if not self.order_manager.has_next:
                break
            self.__static_service.increment_order_num()
            order_occurred_area: Area = self.area_manager.node_id_to_area(self.order_manager.now_order.pick_up_node_id)
            order_occurred_area.orders.append(self.order_manager.now_order)

            if order_occurred_area.num_idle_vehicles or order_occurred_area.num_neighbors:
                MatchedInfo = namedtuple("MatchedInfo", ["matched_vehicle", "road_cost", "order_occurred_area"])
                matched_info = None

                if order_occurred_area.num_idle_vehicles:

                    # Find a nearest car to match the current order
                    # --------------------------------------
                    for vehicle_id in order_occurred_area.get_idle_vehicle_ids():
                        if self.is_train and order_occurred_area.num_idle_vehicles <= 3:
                            continue
                        vehicle = self.vehicle_manager.get_vehicle_by_vehicle_id(vehicle_id)
                        tmp_road_cost = self.__road_cost(
                            start_node_index=self.__node_manager.get_node_index(
                                vehicle.location_node_id
                            ),
                            end_node_index=self.__node_manager.get_node_index(
                                self.order_manager.now_order.pick_up_node_id
                            ),
                        )
                        if matched_info is None:
                            matched_info = MatchedInfo(vehicle, tmp_road_cost, order_occurred_area)
                        elif tmp_road_cost < matched_info.road_cost:
                            matched_info = MatchedInfo(vehicle, tmp_road_cost, order_occurred_area)
                    # --------------------------------------
                # Neighbor car search system to increase search range
                elif self.neighbor_can_server and order_occurred_area.num_neighbors:
                    matched_vehicle, road_cost, order_occurred_area = self.__find_server_vehicle_function(
                        neighbor_server_deep_limit=self.neighbor_server_deep_limit,
                        visit_list={},
                        area=order_occurred_area,
                        tmp_min=None,
                        deep=0,
                    )
                    matched_info = MatchedInfo(matched_vehicle, road_cost, order_occurred_area)
                
                # When all Neighbor Cluster without any idle Vehicles
                if matched_info is None or matched_info.road_cost > self.__pick_up_time_window:
                    self.__static_service.increment_reject_num()
                    self.order_manager.now_order.set_arrive_info(ArriveInfo.REJECT)
                # Successfully matched a vehicle
                else:
                    match_count[order_occurred_area.id] += 1

                    matched_vehicle: Vehicle = matched_info.matched_vehicle
                    road_cost: int = matched_info.road_cost
                    order_occurred_area: Area = matched_info.order_occurred_area
                    self.order_manager.now_order.pick_up_wait_time = road_cost
                    matched_vehicle.orders.append(self.order_manager.now_order)

                    self.__static_service.add_wait_time(
                        self.__road_cost(
                            start_node_index=self.__node_manager.get_node_index(
                                matched_vehicle.location_node_id
                            ),
                            end_node_index=self.__node_manager.get_node_index(
                                self.order_manager.now_order.pick_up_node_id
                            ),
                        )
                    )

                    schedule_cost = self.__road_cost(
                        start_node_index=self.__node_manager.get_node_index(
                            matched_vehicle.location_node_id
                        ),
                        end_node_index=self.__node_manager.get_node_index(
                            self.order_manager.now_order.pick_up_node_id
                        ),
                    ) + self.__road_cost(
                        start_node_index=self.__node_manager.get_node_index(
                            self.order_manager.now_order.pick_up_node_id
                        ),
                        end_node_index=self.__node_manager.get_node_index(
                            self.order_manager.now_order.delivery_node_id
                        ),
                    )

                    # Add a destination to the current vehicle
                    matched_vehicle.delivery_node_id = self.order_manager.now_order.delivery_node_id

                    # Delivery Cluster {Vehicle:ArriveTime}
                    destination_area = self.area_manager.node_id_to_area(self.order_manager.now_order.delivery_node_id)
                    arrival_time = self.real_time_in_experiment + np.timedelta64(schedule_cost * self.__minutes)
                    destination_area.set_arrival_schedule(matched_vehicle.id, arrival_time)

                    # delete now Cluster's recode about now Vehicle
                    order_occurred_area.unregister_idle_vehicle_id(matched_vehicle.id)
                    self.order_manager.now_order.set_arrive_info(ArriveInfo.SUCCESS)
            else:
                # None available idle Vehicles
                self.__static_service.increment_reject_num()
                self.order_manager.now_order.set_arrive_info(ArriveInfo.REJECT)

            # The current order has been processed and start processing the next order
            # ------------------------------
            # breakpoint()
            
            self.order_manager.increment()

        if not self.is_train:
            import os
            import pandas as pd
            path = "./outputs/tmp/match_check.csv"
            df = pd.DataFrame({"GridID": match_count.keys(), "num_matched": match_count.values()})
            df["day"] = end_time.day
            df["hour"] = end_time.hour
            df["minute"] = end_time.minute
            if not os.path.exists(path):
                df.to_csv(path, index=False)
            else:
                tmp = pd.read_csv(path)
                df = pd.concat([tmp, df])
                df.to_csv(path, index=False)

    def get_orders_in_timeslice(self, start_time: datetime, end_time: datetime) -> List[Order]:
        return self.order_manager.get_orders_in_timeslice(start_time, end_time)

    def __find_server_vehicle_function(
        self, neighbor_server_deep_limit, visit_list, area: Area, tmp_min, deep
    ):
        """
        Use dfs visit neighbors and find nearest idle Vehicle
        """
        if deep > neighbor_server_deep_limit or area.id in visit_list:
            return tmp_min

        visit_list[area.id] = True
        for vehicle_id in area.get_idle_vehicle_ids():
            vehicle = self.vehicle_manager.get_vehicle_by_vehicle_id(vehicle_id)
            tmp_road_cost = self.__road_cost(
                start_node_index=self.__node_manager.get_node_index(
                    vehicle.location_node_id
                ),
                end_node_index=self.__node_manager.get_node_index(
                    self.order_manager.now_order.pick_up_node_id
                ),
            )
            if tmp_min == None:
                tmp_min = (vehicle, tmp_road_cost, area)
            elif tmp_road_cost < tmp_min[1]:
                tmp_min = (vehicle, tmp_road_cost, area)

        if self.neighbor_can_server:
            for neighbor_area_id in area.get_neighbor_ids():
                neighbor_area = self.area_manager.get_area_by_area_id(neighbor_area_id)
                tmp_min = self.__find_server_vehicle_function(
                    neighbor_server_deep_limit,
                    visit_list,
                    neighbor_area,
                    tmp_min,
                    deep + 1,
                )
        return tmp_min

    def update(self) -> None:
        self.__update_function()

    def __update_function(self) -> None:
        """
        - Vehicles move to destination area.
        - Areas register arrived vehicle as idle vehicle.
        """
        # move vehicles have passengers.
        for area in self.area_manager.get_area_list():
            area.orders.clear()
            for schedule in area.get_arrival_schedules():
                if schedule.arrival_time <= self.real_time_in_experiment:
                    vehicle = self.vehicle_manager.get_vehicle_by_vehicle_id(schedule.vehicle_id)
                    vehicle.move(destination_area_id=area.id)
                    area.register_vehicle_id_as_idle_status(vehicle.id)
                    area.remove_schedule(schedule)
        # mode idle vehicles.
        for dispatch_order in self.dispatch_order_list:
            if dispatch_order.start_node_id != dispatch_order.end_node_id:
                vehicle = self.vehicle_manager.get_vehicle_by_vehicle_id(dispatch_order.vehicle_id)
                start_area = self.area_manager.node_id_to_area(dispatch_order.start_node_id)
                dispatch_area = self.area_manager.node_id_to_area(dispatch_order.end_node_id)
                vehicle.deploy_to_node(dispatch_order.end_node_id)
                vehicle.deploy_to_area(dispatch_area.id)
                dispatch_area.register_vehicle_id_as_idle_status(vehicle.id)
                start_area.unregister_idle_vehicle_id(vehicle.id)
                self.__static_service.increment_dispatch_num()
                dispatch_cost = self.__road_cost(
                    self.__node_manager.get_node_index(dispatch_order.start_node_id),
                    self.__node_manager.get_node_index(dispatch_order.end_node_id),
                )
                self.__static_service.add_dispatch_cost(dispatch_cost)

    def proceed(self) -> None:
        self.step += 1
        self.real_time_in_experiment += self.time_periods

    def count_idle_vehicles(self) -> None:
        for area in self.area_manager.get_area_list():
            area.per_dispatch_idle_vehicles = area.num_idle_vehicles

    @property
    def end_time(self) -> datetime.datetime:
        return self.order_manager.last_order_start_time

    def init_time(self):
        self.real_time_in_experiment = self.order_manager.farst_order_start_time
        self.step = 0

    def __call__(self) -> None:
        self.init_time()
        end_time = self.end_time

        __episode_start_time = datetime.datetime.now()
        print("Start experiment")
        print("----------------------------")
        while self.real_time_in_experiment < end_time:
            # __step_update_start_time = datetime.datetime.now()
            self.update()
            # self.__static_service.add_update_time(
            #     datetime.datetime.now() - __step_update_start_time
            # )

            # __step_match_start_time = datetime.datetime.now()
            self.match()
            # self.__static_service.add_match_time(
            #     datetime.datetime.now() - __step_match_start_time
            # )

            # __step_reward_start_time = datetime.datetime.now()
            # reward_array = self.__reward_function()
            # self.__static_service.add_reward_time(
            #     datetime.datetime.now() - __step_reward_start_time
            # )

            # __step_next_state_start_time = datetime.datetime.now()
            # self.__get_next_state_function()
            # self.__static_service.add_next_state_time(
            #     datetime.datetime.now() - __step_next_state_start_time
            # )
            # for area in self.area_manager.get_area_list():
            #     area.dispatch_number = 0

            # __step_learning_start_time = datetime.datetime.now()
            # self.__learning_function()
            # self.__static_service.add_learning_time(
            #     datetime.datetime.now() - __step_learning_start_time
            # )

            # __step_demand_predict_start_time = datetime.datetime.now()
            # self.__demand_predict_function(
            #     start_datetime=self.real_time_in_experiment,
            #     end_datetime=self.real_time_in_experiment + self.time_periods,
            #     feature=None,
            #     num_areas=self.num_areas,
            # )
            # self.__supply_expect_function()
            # self.__static_service.add_demand_predict_time(
            #     datetime.datetime.now() - __step_demand_predict_start_time
            # )

            # Count the number of idle vehicles before Dispatch
            self.count_idle_vehicles()

            # step_dispatch_start_time = datetime.datetime.now()
            # self.__dispatch_function()
            # self.__static_service.add_dispatch_time(
            #     datetime.datetime.now() - step_dispatch_start_time
            # )
            # Count the number of idle vehicles after Dispatch
            for area in self.area_manager.get_area_list():
                area.later_dispatch_idle_vehicles = area.num_idle_vehicles

            # A time slot is processed
            self.proceed()
            
        # ------------------------------------------------
        __episode_end_time = datetime.datetime.now()

        sum_order_value = 0
        order_value_num = 0
        for order in self.order_manager.get_orders():
            if order.arrive_info == ArriveInfo.SUCCESS:
                sum_order_value += order.order_value
                order_value_num += 1

        self.print_stats()
        self.__static_service.save_stats(
            date=f"2016{self.order_manager.farst_order_start_time.month}{self.order_manager.farst_order_start_time.day}"
        )

    def save_stats(self) -> None:
        self.__static_service.save_stats(
            date=f"2016{self.order_manager.farst_order_start_time.month}{self.order_manager.farst_order_start_time.day}"
        )

    def write_stats(
        self,
        data_size: str,
        num_vehicles: int,
        area_mode: AreaMode,
        dispatch_mode: DispatchMode
    ) -> None:
        self.__static_service.write_stats(
            data_size=data_size,
            num_vehicles=num_vehicles,
            area_mode=area_mode,
            dispatch_mode=dispatch_mode,
        )

    def save_dispatch_history(self, dispatch_order_list: List[DispatchOrder]) -> None:
        dispatch_data = np.array(
            [
                [
                    self.real_time_in_experiment,
                    dispatch_order.vehicle_id,
                    dispatch_order.from_area_id,
                    dispatch_order.to_area_id,
                ]
                for dispatch_order in dispatch_order_list
            ]
        )
        self.__static_service.save_dispatch_history(dispatch_data)

    def write_dispatch_history(self) -> None:
        self.__static_service.write_dispatch_history()

    def __workday_or_weekend(self, day) -> str:
        if type(day) != type(0) or day < 0 or day > 6:
            raise Exception("input format error")
        elif day == 5 or day == 6:
            return "Weekend"
        else:
            return "Workday"

    def print_stats(self) -> None:
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
        print("Number of Vehicles: " + str(self.vehicles_number))
        print("Number of Orders: " + str(len(self.order_manager)))
        print("Number of Reject: " + str(self.__static_service.reject_num))
        print("Number of Dispatch: " + str(self.__static_service.dispatch_num))
        if (self.__static_service.dispatch_num) != 0:
            print(
                "Average Dispatch Cost: "
                + str(
                    self.__static_service.totally_dispatch_cost
                    / self.__static_service.dispatch_num
                )
            )
        if (len(self.order_manager) - self.__static_service.reject_num) != 0:
            print(
                "Average wait time: "
                + str(
                    self.__static_service.totally_wait_time
                    / (len(self.order_manager) - self.__static_service.reject_num)
                )
            )
        # print("Totally Order value: " + str(sum_order_value))
        print("Totally Update Time : " + str(self.__static_service.totally_update_time))
        print(
            "Totally NextState Time : "
            + str(self.__static_service.totally_next_state_time)
        )
        print(
            "Totally Learning Time : " + str(self.__static_service.totally_learning_time)
        )
        print(
            "Totally Demand Predict Time : "
            + str(self.__static_service.totally_demand_predict_time)
        )
        print(
            "Totally Dispatch Time : " + str(self.__static_service.totally_dispatch_time)
        )
        print(
            "Totally Simulation Time : " + str(self.__static_service.totally_match_time)
        )
        # print("Episode Run time : " + str(__episode_end_time - __episode_start_time))
