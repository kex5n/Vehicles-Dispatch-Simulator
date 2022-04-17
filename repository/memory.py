from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np


@dataclass(frozen=True)
class Record:
    vehicle_id: int
    time: datetime
    from_area_id: int
    to_area_id: int
    state: np.ndarray


class Memory:
    def __init__(self):
        self.__records: List[Record] = []
        self.__count = 0

    def save(
        self,
        vehicle_id: int,
        time: datetime,
        from_area_id: int,
        to_area_id: int,
        state:np.ndarray,
    ) -> None:
        record = Record(
            vehicle_id=vehicle_id,
            time=time,
            from_area_id=from_area_id,
            to_area_id=to_area_id,
            state=deepcopy(state),
        )
        self.__records.append(record)
        self.__count += 1

    def __refresh(self) -> None:
        self.__records.clear()

    def pop(self) -> List[Record]:
        return_value = deepcopy(self.__records)
        self.__refresh()
        return return_value

    @property
    def has_saved(self) -> bool:
        return self.__count > 0
