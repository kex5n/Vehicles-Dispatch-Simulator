from datetime import datetime, timedelta
from math import asin, cos, radians, sin, sqrt

import numpy as np

from domain.demand_prediction_mode import DemandPredictionMode


def haversine(lon1, lat1, lon2, lat2) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


class DataModule:
    def __init__(self, demand_prediction_mode: DemandPredictionMode):
        self.__demand_prediction_mode = demand_prediction_mode
        if self.__demand_prediction_mode == DemandPredictionMode.TRAIN:
            self.__date = datetime(2016, 6, 1)
        else:
            self.__date = datetime(2016, 6, 24)

    @property
    def date(self) -> str:
        return f"{self.__date.month}".zfill(2) + f"{self.__date.day}".zfill(2)

    def next(self) -> bool:
        next_day = self.__date + timedelta(days=1)
        self.__date = next_day
        if self.__demand_prediction_mode == DemandPredictionMode.TRAIN:
            if next_day.day == 24:
                return False
        else:
            if next_day.month != 6:
                return False
        return True

    def __str__(self) -> str:
        return self.date
