import datetime
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd

from domain import AreaMode, DispatchMode
from modules.dispatch import DispatchOrder


@dataclass
class StaticsService:
    def __init__(self):
        self.__experiment_start_time = datetime.datetime.now()
        self.__output_dir = (
            Path(__file__).parents[1] / "outputs"
            / f"{self.__experiment_start_time.year}-{self.__experiment_start_time.month}-{self.__experiment_start_time.day}_{self.__experiment_start_time.hour}:{self.__experiment_start_time.minute}"
        )
        self.__stats_df: pd.DataFrame = pd.DataFrame(
            columns=[
                "date",
                "order_num",
                "reject_num",
                "dispatch_num",
                "totally_dispatch_cost",
                "totally_wait_time",
            ]
        )
        self.__dispatch_history_df: pd.DataFrame = pd.DataFrame(
            columns = [
                "datetime",
                "vehicle_id",
                "from_area_id",
                "to_area_id",
            ]
        )
        self.__order_num: int = 0
        self.__reject_num: int = 0
        self.__dispatch_num: int = 0
        self.__totally_dispatch_cost: int = 0
        self.__totally_wait_time: int = 0
        self.__totally_update_time = datetime.timedelta()
        self.__totally_reward_time = datetime.timedelta()
        self.__totally_next_state_time = datetime.timedelta()
        self.__totally_learning_time = datetime.timedelta()
        self.__totally_dispatch_time = datetime.timedelta()
        self.__totally_match_time = datetime.timedelta()
        self.__totally_demand_predict_time = datetime.timedelta()

    def reset(self):
        self.__order_num: int = 0
        self.__reject_num: int = 0
        self.__dispatch_num: int = 0
        self.__totally_dispatch_cost: int = 0
        self.__totally_wait_time: int = 0
        self.__totally_update_time = datetime.timedelta()
        self.__totally_reward_time = datetime.timedelta()
        self.__totally_next_state_time = datetime.timedelta()
        self.__totally_learning_time = datetime.timedelta()
        self.__totally_dispatch_time = datetime.timedelta()
        self.__totally_match_time = datetime.timedelta()
        self.__totally_demand_predict_time = datetime.timedelta()

    def increment_order_num(self) -> None:
        self.__order_num += 1

    @property
    def reject_num(self) -> int:
        return self.__reject_num

    def increment_reject_num(self) -> None:
        self.__reject_num += 1

    @property
    def dispatch_num(self) -> int:
        return self.__dispatch_num

    def increment_dispatch_num(self) -> None:
        self.__dispatch_num += 1

    @property
    def totally_dispatch_cost(self) -> int:
        return self.__totally_dispatch_cost

    def add_dispatch_cost(self, dispatch_cost: int) -> None:
        self.__totally_dispatch_cost += dispatch_cost

    @property
    def totally_update_time(self) -> datetime.timedelta:
        return self.__totally_update_time

    def add_update_time(self, delta: datetime.timedelta) -> None:
        self.__totally_update_time += delta

    @property
    def totally_match_time(self) -> datetime.timedelta:
        return self.__totally_match_time

    def add_match_time(self, delta: datetime.timedelta) -> None:
        self.__totally_match_time += delta

    @property
    def totally_reward_time(self) -> datetime.timedelta:
        return self.__totally_reward_time

    def add_reward_time(self, delta: datetime.timedelta) -> None:
        self.__totally_reward_time += delta

    @property
    def totally_next_state_time(self) -> datetime.timedelta:
        return self.__totally_next_state_time

    def add_next_state_time(self, delta: datetime.timedelta) -> None:
        self.__totally_next_state_time += delta

    @property
    def totally_learning_time(self) -> datetime.timedelta:
        return self.__totally_learning_time

    def add_learning_time(self, delta: datetime.timedelta) -> None:
        self.__totally_learning_time += delta

    @property
    def totally_demand_predict_time(self) -> datetime.timedelta:
        return self.__totally_demand_predict_time

    def add_demand_predict_time(self, delta: datetime.timedelta) -> None:
        self.__totally_demand_predict_time += delta

    @property
    def totally_dispatch_time(self) -> datetime.timedelta:
        return self.__totally_dispatch_time

    def add_dispatch_time(self, delta: datetime.timedelta) -> None:
        self.__totally_dispatch_time += delta

    @property
    def totally_wait_time(self) -> int:
        return self.__totally_wait_time

    def add_wait_time(self, cost: int) -> None:
        self.__totally_wait_time += cost

    def save_stats(self, date: str) -> None:
        tmp_df = pd.DataFrame(
            {
                "date": [date],
                "order_num": [self.__order_num],
                "reject_num": [self.__reject_num],
                "dispatch_num": [self.__dispatch_num],
                "totally_dispatch_cost": [self.__totally_dispatch_cost],
                "totally_wait_time": [self.__totally_wait_time],
            }
        )
        self.__stats_df = pd.concat([self.__stats_df, tmp_df])
        

    def write_stats(
        self,
        data_size: str,
        num_vehicles: int,
        area_mode: AreaMode,
        dispatch_mode: DispatchMode
    ) -> None:
        os.makedirs(self.__output_dir, exist_ok=True)
        self.__stats_df["data_size"] = data_size
        self.__stats_df["num_vehicles"] = num_vehicles
        self.__stats_df["area_mode"] = area_mode.value
        self.__stats_df["dispatch_mode"] = dispatch_mode.value
        self.__stats_df.to_csv(self.__output_dir / f"statistics.csv", index=False)

    def save_dispatch_history(self, dispatch_data: np.ndarray) -> None:
        if len(dispatch_data) == 0:
            return
        df = pd.DataFrame(dispatch_data, columns=self.__dispatch_history_df.columns)
        self.__dispatch_history_df = pd.concat([self.__dispatch_history_df, df])

    def write_dispatch_history(self) -> None:
        os.makedirs(self.__output_dir, exist_ok=True)
        self.__dispatch_history_df.to_csv(self.__output_dir / f"dispatch_history.csv", index=False)
