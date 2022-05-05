from dataclasses import dataclass


# random.seed(1234)
# np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)
# torch.backends.cudnn.deterministic = True

@dataclass(frozen=True)
class LocalRegionBound:
    west_bound: float
    east_bound: float
    north_bound: float
    south_bound: float

    def __str__(self):
        return f"{self.west_bound}, {self.east_bound}, {self.south_bound}, {self.north_bound}"
