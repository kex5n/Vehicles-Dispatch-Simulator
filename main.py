from config import Config
from domain.demand_prediction_mode import DemandPredictionMode
from simulator.simulator import Simulation
from util import DataModule

if __name__ == "__main__":
    config = Config.load()
    simulator = Simulation(
        area_mode=config.AREA_MODE,
        demand_prediction_mode=config.DEMAND_PREDICTION_MODE,
        dispatch_mode=config.DISPATCH_MODE,
        vehicles_number=config.VEHICLES_NUMBER,
        time_periods=config.TIMESTEP,
        local_region_bound=config.LOCAL_REGION_BOUND,
        side_length_meter=config.SIDE_LENGTH_KIRO_METER,
        vehicles_server_meter=config.VEHICLE_SERVICE_KIRO_METER,
        neighbor_can_server=config.NELGHBOR_CAN_SERVER,
        minutes=config.MINUTES,
        pick_up_time_window=config.PICKUPTIMEWINDOW
    )

    date_module = DataModule(demand_prediction_mode=config.DEMAND_PREDICTION_MODE)
    simulator.create_all_instantiate(date_module.date)
    simulator()

    while date_module.next():
        simulator.reload(date_module.date)
        simulator()

    if config.DEMAND_PREDICTION_MODE == DemandPredictionMode.TEST:
        simulator.static_service.write_stats(
            data_size=config.DATA_SIZE,
            num_vehicles=config.VEHICLES_NUMBER,
            area_mode=config.AREA_MODE,
            dispatch_mode=config.DISPATCH_MODE,
        )
