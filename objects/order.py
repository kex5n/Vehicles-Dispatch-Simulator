from datetime import datetime
from typing import List

import pandas as pd

from domain.arrive_info import ArriveInfo


class Order(object):
    def __init__(
        self,
        id: int,
        order_time: datetime,
        pick_up_node_id: int,
        delivery_node_id: int,
        pick_up_time_window,
        pick_up_wait_time,
        arrive_info: ArriveInfo,
        order_value: int,
    ):
        self.id: int = id  # This order's ID
        self.order_time: datetime = order_time  # Start time of this order
        self.pick_up_node_id: int = (
            pick_up_node_id  # The starting position of this order
        )
        self.delivery_node_id: int = delivery_node_id  # Destination of this order
        self.pick_up_time_window = (
            pick_up_time_window  # Limit of waiting time for this order
        )
        self.pick_up_wait_time = pick_up_wait_time  # This order's real waiting time from running in the simulator
        self.__arrive_info: ArriveInfo = (
            arrive_info  # Processing information for this order
        )
        self.order_value = order_value  # The value of this order

    @property
    def arrive_info(self) -> ArriveInfo:
        return self.__arrive_info

    def set_arrive_info(self, arrive_info: ArriveInfo) -> None:
        if not isinstance(arrive_info, ArriveInfo):
            raise ValueError
        self.__arrive_info = arrive_info

    def arrive_order_time_record(self, arrive_time) -> None:
        self.arrive_info = "ArriveTime:" + str(arrive_time)

    def example(self) -> None:
        print("Order Example output")
        print("ID:", self.id)
        print("ReleasTime:", self.order_time)
        print("PickupPoint:", self.pick_up_node_id)
        print("DeliveryPoint:", self.delivery_node_id)
        print("PickupTimeWindow:", self.pick_up_time_window)
        print("PickupWaitTime:", self.pick_up_wait_time)
        print("ArriveInfo:", self.arrive_info)
        print()

    def reset(self) -> None:
        self.pick_up_wait_time = None
        self.arrive_info = None


class OrderManager:
    def __init__(self, order_df: pd.DataFrame, pick_up_time_window):
        self.__order_list: List[Order] = [
            Order(
                id=order_row["ID"],
                order_time=order_row["Start_time"],
                pick_up_node_id=order_row["NodeS"],
                delivery_node_id=order_row["NodeE"],
                pick_up_time_window=order_row["Start_time"] + pick_up_time_window,
                pick_up_wait_time=None,
                arrive_info=None,
                order_value=None,
            )
            for _, order_row in order_df.iterrows()
        ]
        self.__order_index = 0

    @property
    def now_order(self) -> Order:
        return self.__order_list[self.__order_index]

    def increment(self) -> None:
        self.__order_index += 1

    def get_orders(self) -> List[Order]:
        return [order for order in self.__order_list]

    def set_order_cost(self, idx: int, cost: int) -> None:
        self.__order_list[idx].order_value = cost

    def reset(self):
        self.__order_list = None
        self.__order_index = 0

    def reload(self, order_df: pd.DataFrame, pick_up_time_window) -> None:
        self.__order_list: List[Order] = [
            Order(
                id=idx,
                order_time=order_row["Start_time"],
                pick_up_node_id=order_row["NodeS"],
                delivery_node_id=order_row["NodeE"],
                pick_up_time_window=order_row["Start_time"] + pick_up_time_window,
                pick_up_wait_time=None,
                arrive_info=None,
                order_value=None,
            )
            for idx, order_row in order_df.iterrows()
        ]
        self.__order_index = 0

    def __len__(self) -> int:
        return len(self.__order_list)

    @property
    def has_next(self) -> bool:
        return self.__order_index + 1 != len(self.__order_list)

    @property
    def farst_order_start_time(self) -> datetime:
        return self.__order_list[0].order_time

    @property
    def last_order_start_time(self) -> datetime:
        return self.__order_list[-1].order_time
