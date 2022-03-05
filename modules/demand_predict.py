from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from domain import AreaMode

from domain.demand_prediction_mode import DemandPredictionMode


class DemandPredictorInterface:
    def predict(self, start_datetime: datetime, end_datetime: datetime, feature: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MockDemandPredictor(DemandPredictorInterface):
    def __init__(self, demand_prediction_mode: DemandPredictionMode, area_mode: AreaMode):
        self.__demand_predict_mode: DemandPredictionMode = demand_prediction_mode
        self.__area_mode: AreaMode = area_mode
        self.__date: datetime = None
        self.__order_df = None

    def predict(self, start_datetime: datetime, end_datetime: datetime, feature: np.ndarray, num_areas: int) -> np.ndarray:
        if (self.__date is None) or (not self.__is_same_date(end_datetime)):
            file_name = f"order_2016{str(end_datetime.month).zfill(2)}{str(end_datetime.day).zfill(2)}.csv"
            order_path = Path(__file__).parents[1] / "data" / "Order" / "modified" / "dummy" / file_name
            self.__order_df = pd.read_csv(order_path)
            self.__date = end_datetime.date()
        start_timestamp = int(start_datetime.timestamp())
        end_timestamp = int(end_datetime.timestamp())
        df = self.__order_df[
            (self.__order_df["Start_time"]>=start_timestamp) & (self.__order_df["End_time"]<=end_timestamp)
        ]
        if self.__area_mode == AreaMode.TRANSPORTATION_CLUSTERING:
            pred_array = []
            summary_df = df.groupby("Start_GridID").count().reset_index()[["Start_GridID", "ID"]].sort_values("Start_GridID")
            for i in range(num_areas):
                tmp_df = summary_df[summary_df["Start_GridID"]==i]
                if len(tmp_df) == 0:
                    pred_array.append(0)
                else:
                    pred_array.append(tmp_df["ID"].values[0])
            return np.array(pred_array)
        else:
            raise NotImplementedError

    def __is_same_date(self, other: datetime):
        return self.__date == other.date()