from config import Config
from modules.dispatch import load_dispatch_component, DispatchModuleInterface
from simulator.simulator import Simulator
from util import DataModule

if __name__ == "__main__":
    config = Config.load()
    simulator = Simulator(
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
    demand_predictor_module: DispatchModuleInterface = load_dispatch_component(config.DISPATCH_MODE)
    date_module = DataModule(demand_prediction_mode=config.DEMAND_PREDICTION_MODE)
    simulator.create_all_instantiate(date_module.date)

    while date_module.next():
        simulator.init_time()
        end_time = simulator.end_time
        while simulator.real_time_in_experiment <= end_time:
            simulator.update()
            simulator.match()
            simulator.count_idle_vehicles()
            area_manager = simulator.area_manager_copy
            vehicle_manager = simulator.vehicle_manager_copy
            dispatch_order_list = demand_predictor_module(
                area_manager=area_manager,
                vehicle_manager=vehicle_manager
            )
            simulator.set_dispatch_orders(dispatch_order_list)
            simulator.proceed()
        simulator.print_stats()
        simulator.reload(date_module.date)
