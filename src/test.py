from datetime import datetime
from pathlib import Path
import warnings
from domain.demand_prediction_mode import DemandPredictionMode

from domain.dispatch_mode import DispatchMode
warnings.simplefilter('ignore')

from config import Config
from modules import (
    DemandPredictorInterface,
    RewardCalculator,
    load_demand_prediction_component,
    load_dispatch_component,
)
from modules.dispatch import DQNDispatch
from modules.state import FeatureManager
from simulator.simulator import Simulator
from util import DataModule


# random.seed(1234)
# np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)
# torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    config = Config.load()
    demand_prediction_mode = DemandPredictionMode.TEST
    simulate_start_time = datetime.now()
    dqn_checkpoint_path = (
        Path(__file__).parents[1] / "models" / "checkpoints" / "dqn" 
        / "2022-5-5_18:29:14" / "episode2.cpt"
    )
    # init modules
    simulator = Simulator(
        area_mode=config.AREA_MODE,
        demand_prediction_mode=demand_prediction_mode,
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
    reward_culcurator = RewardCalculator()
    demand_predictor: DemandPredictorInterface = load_demand_prediction_component(
        dispatch_mode=config.DISPATCH_MODE,
        demand_prediction_mode=demand_prediction_mode,
        area_mode=config.AREA_MODE,
    )
    feature_manager = FeatureManager(k=5)
    dispatch_module: DQNDispatch = load_dispatch_component(
        dispatch_mode=config.DISPATCH_MODE,
        config=config,
    )
    if config.DISPATCH_MODE == DispatchMode.DQN:
        dispatch_module.load(dqn_checkpoint_path)

    print(f"=========================== test start ===========================")
    date_module = DataModule(demand_prediction_mode=demand_prediction_mode)
    simulator.create_all_instantiate(date_module.date)
    while date_module.next():
        simulator.init_time()
        end_time = simulator.end_time
        start_datetime_of_this_timeslice = simulator.real_time_in_experiment
        end_datetime_of_this_timeslice = simulator.real_time_in_experiment + simulator.time_periods

        # set initial dispatch order
        area_manager = simulator.area_manager_copy
        vehicle_manager = simulator.vehicle_manager_copy
        order_manager = simulator.order_manager_copy

        prediction = demand_predictor.predict(
            start_datetime=start_datetime_of_this_timeslice,
            end_datetime=end_datetime_of_this_timeslice,
            feature=None,
            num_areas=area_manager.num_areas,
            debug=config.DEBUG
        )
        dispatch_order_list = dispatch_module(
            area_manager=area_manager,
            vehicle_manager=vehicle_manager,
            prediction=prediction,
        )
        simulator.set_dispatch_orders(dispatch_order_list)
        simulator.save_dispatch_history(dispatch_order_list)

        while simulator.real_time_in_experiment < end_time:
            start_datetime_of_this_timeslice = simulator.real_time_in_experiment
            end_datetime_of_this_timeslice = simulator.real_time_in_experiment + simulator.time_periods

            # step simulator
            simulator.update()
            # for debug
            area_manager = simulator.area_manager_copy
            before = [area.num_idle_vehicles for area in area_manager.get_area_list()]

            simulator.match()
            simulator.count_idle_vehicles()
            
            # for debug
            area_manager = simulator.area_manager_copy
            after = [area.num_idle_vehicles for area in area_manager.get_area_list()]
            # if (after[4] != 0):
            #     breakpoint()
            
            simulator.count_idle_vehicles()

            area_manager = simulator.area_manager_copy
            vehicle_manager = simulator.vehicle_manager_copy

            # ===================== culculate next dispatch =====================
            prediction = demand_predictor.predict(
                start_datetime=start_datetime_of_this_timeslice,
                end_datetime=end_datetime_of_this_timeslice,
                feature=None,
                num_areas=area_manager.num_areas,
                debug=config.DEBUG
            )

            # plan dispatch order
            dispatch_order_list = dispatch_module(
                area_manager=area_manager,
                vehicle_manager=vehicle_manager,
                prediction=prediction,
            )

            # set dispatch order to simulator
            simulator.set_dispatch_orders(dispatch_order_list)
            # if simulator.real_time_in_experiment.hour == 12:
            #     breakpoint()
            simulator.save_dispatch_history(dispatch_order_list)
            # =================================================================

            # proceed to next timeslice
            simulator.proceed()

        # finish simulation of the day
        # simulator.print_stats()
        simulator.save_stats()
        # proceed to next day
        simulator.reload(date_module.date)

    simulator.write_stats(
        data_size=config.DATA_SIZE,
        num_vehicles=config.VEHICLES_NUMBER,
        area_mode=config.AREA_MODE,
        dispatch_mode=config.DISPATCH_MODE,
    )
    simulator.write_dispatch_history()
