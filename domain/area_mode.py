from enum import Enum


class AreaMode(Enum):
    GRID = "Grid"
    TRANSPORTATION_CLUSTERING = "TransportationClustering"
    K_MEANS_CLUSTERING = "KmeansClustering"
    SPECTRAL_CLUSTERING = "SpectralClustering"
