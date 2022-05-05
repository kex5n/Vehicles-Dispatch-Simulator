from enum import Enum


# random.seed(1234)
# np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)
# torch.backends.cudnn.deterministic = True

class AreaMode(Enum):
    GRID = "Grid"
    TRANSPORTATION_CLUSTERING = "TransportationClustering"
    K_MEANS_CLUSTERING = "KmeansClustering"
    SPECTRAL_CLUSTERING = "SpectralClustering"
