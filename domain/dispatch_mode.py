from enum import Enum


class DispatchMode(Enum):
    DQN: str = "DQN"
    RANDOM: str = "Random"
    NOT_DISPATCH: str = "NotDispatch"
