from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from domain import AreaMode, DispatchMode

from domain.demand_prediction_mode import DemandPredictionMode

# random.seed(1234)
np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)
# torch.backends.cudnn.deterministic = True


class DemandPredictorInterface:
    def predict(self, start_datetime: datetime, end_datetime: datetime, feature: np.ndarray, num_areas: int) -> np.ndarray:
        raise NotImplementedError

class RandomDemandPredictor(DemandPredictorInterface):
    def predict(self, start_datetime: datetime, end_datetime: datetime, feature: np.ndarray, num_areas: int) -> np.ndarray:
        return np.zeros(num_areas)

class MockDemandPredictor(DemandPredictorInterface):
    def __init__(self, demand_prediction_mode: DemandPredictionMode, area_mode: AreaMode):
        self.__demand_predict_mode: DemandPredictionMode = demand_prediction_mode
        self.__area_mode: AreaMode = area_mode
        self.__date: datetime = None
        self.__order_df = None

    def predict(self, start_datetime: datetime, end_datetime: datetime, feature: np.ndarray, num_areas: int, debug: bool = False) -> np.ndarray:
        if (self.__date is None) or (not self.__is_same_date(end_datetime)):
            file_name = f"order_2016{str(end_datetime.month).zfill(2)}{str(end_datetime.day).zfill(2)}.csv"
            if not debug:
                order_path = Path(__file__).parent / "dummy_data" / file_name
            else:
                order_path = Path(__file__).parents[1] / "data" / "Order" / "modified" / "dummy" / file_name
            # breakpoint()
            self.__order_df = pd.read_csv(order_path)
            self.__date = end_datetime.date()
        start_timestamp = int(start_datetime.timestamp())
        end_timestamp = int(end_datetime.timestamp())
        df = self.__order_df[
            (self.__order_df["Start_time"]>=start_timestamp) & (self.__order_df["End_time"]<=end_timestamp)
        ]
        if self.__area_mode == AreaMode.TRANSPORTATION_CLUSTERING or debug:
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

class TrainingDemandPredictor(DemandPredictorInterface):
    def __init__(self):
        self.order_df = pd.read_csv("./data/medium/Order/modified/train/train.csv")

    def predict(self, start_datetime: datetime, end_datetime: datetime, feature: np.ndarray, num_areas: int, debug: bool = False) -> np.ndarray:
        df = self.order_df[(self.order_df["day"]==end_datetime.day) & (self.order_df["hour"]==end_datetime.hour) & (self.order_df["minute"]==end_datetime.minute)]
        df.reset_index(inplace=True)
        df.sort_values("GridID", inplace=True)
        return np.array(df["target"].values)

class GCNDemandPredictor(DemandPredictorInterface):
    def __init__(self):
        self.order_df = pd.read_csv("./models/checkpoints/gcn/pred.csv")

    def predict(self, start_datetime: datetime, end_datetime: datetime, feature: np.ndarray, num_areas: int, debug: bool = False) -> np.ndarray:
        df = self.order_df[(self.order_df["day"]==end_datetime.day) & (self.order_df["hour"]==end_datetime.hour) & (self.order_df["minute"]==end_datetime.minute)]
        df.reset_index(inplace=True)
        df.sort_values("GridID", inplace=True)
        return np.array(df["pred"].values)

def load_demand_prediction_component(
    dispatch_mode: DispatchMode,
    demand_prediction_mode: DemandPredictionMode,
    area_mode: AreaMode,
    debug: bool = False
) -> DemandPredictorInterface:
    if dispatch_mode == dispatch_mode.RANDOM:
        return RandomDemandPredictor()
    if demand_prediction_mode == DemandPredictionMode.TRAIN:
        return TrainingDemandPredictor()
    if demand_prediction_mode == DemandPredictionMode.TEST:
        return GCNDemandPredictor()
    return MockDemandPredictor(
        demand_prediction_mode=demand_prediction_mode,
        area_mode=area_mode,
    )
