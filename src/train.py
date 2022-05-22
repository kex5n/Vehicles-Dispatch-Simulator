from datetime import datetime
from pathlib import Path
import os
import numpy as np
import warnings

from domain.demand_prediction_mode import DemandPredictionMode
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
np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)
# torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    config = Config.load()
    demand_prediction_mode = DemandPredictionMode.TRAIN
    simulate_start_time = datetime.now()
    dqn_checkpoint_dir = (
        Path(__file__).parents[1] / "models" / "checkpoints" / "dqn" 
        / f"{simulate_start_time.year}-{simulate_start_time.month}-{simulate_start_time.day}_{simulate_start_time.hour}:{simulate_start_time.minute}:{simulate_start_time.second}"
    )
    os.makedirs(dqn_checkpoint_dir)

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
        pick_up_time_window=config.PICKUPTIMEWINDOW,
        data_size=config.DATA_SIZE,
        debug_=config.DEBUG,
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
        is_train=True,
    )

    date_module = DataModule(demand_prediction_mode=DemandPredictionMode.TRAIN)
    simulator.create_all_instantiate(date_module.date)
    for episode in range(config.EPISODE):
        print(f"=========================== EPISODE {episode+1}: start ===========================")
        while date_module.next():
            simulator.init_time()
            end_time = simulator.end_time
            start_datetime_of_this_timeslice = simulator.real_time_in_experiment
            end_datetime_of_this_timeslice = simulator.real_time_in_experiment + simulator.time_periods

            # set initial dispatch order
            area_manager = simulator.area_manager
            vehicle_manager = simulator.vehicle_manager

            prediction = demand_predictor.predict(
                start_datetime=start_datetime_of_this_timeslice,
                end_datetime=end_datetime_of_this_timeslice,
                feature=None,
                num_areas=area_manager.num_areas,
                debug=config.DEBUG,
            )
            dispatch_order_list = dispatch_module(
                area_manager=area_manager,
                vehicle_manager=vehicle_manager,
                prediction=prediction,
            )
            for dispatch_order in dispatch_order_list:
                feature_manager.register_action(vehicle_id=dispatch_order.vehicle_id, action=dispatch_order.action)
                from_area_id = area_manager.node_id_to_area(node_id=dispatch_order.start_node_id).id
                feature_manager.register_from_area_id(vehicle_id=dispatch_order.vehicle_id, from_area_id=from_area_id)
                to_area_id = area_manager.node_id_to_area(node_id=dispatch_order.end_node_id).id
                feature_manager.register_to_area_id(vehicle_id=dispatch_order.vehicle_id, to_area_id=to_area_id)
            simulator.set_dispatch_orders(dispatch_order_list)

            while simulator.real_time_in_experiment < end_time:
                start_datetime_of_this_timeslice = simulator.real_time_in_experiment
                end_datetime_of_this_timeslice = simulator.real_time_in_experiment + simulator.time_periods
                # ===================== calculate current state =====================
                area_manager = simulator.area_manager
                vehicle_manager = simulator.vehicle_manager

                # summarize supply
                supply_array = np.array([np.float32(area.num_idle_vehicles) for area in area_manager.get_area_list()])

                # culculate state
                for vehicle in vehicle_manager.get_dispatched_vehicle_list():
                    area = area_manager.get_area_by_area_id(vehicle.location_area_id)
                    state = feature_manager.calc_state(area=area, demand_array=prediction, supply_array=supply_array)
                    feature_manager.register_state(vehicle_id=vehicle.id, state_array=state)
                # ===================================================================

                # step simulator
                simulator.update()
                area_manager = simulator.area_manager
                middle_num_idle_vehicles = [area.num_idle_vehicles for area in area_manager.get_area_list()]
                simulator.match()
                simulator.count_idle_vehicles()

                area_manager = simulator.area_manager
                vehicle_manager = simulator.vehicle_manager

                # ===================== calculate next state =====================
                order_list = simulator.get_orders_in_timeslice(
                    start_time=start_datetime_of_this_timeslice,
                    end_time=end_datetime_of_this_timeslice,
                )

                demand_array = np.zeros(area_manager.num_areas)
                for order in order_list:
                    area_id = area_manager.node_id_to_area(order.pick_up_node_id).id
                    demand_array[area_id] += 1

                # summarize supply
                supply_array = np.array([np.float32(area.num_idle_vehicles) for area in area_manager.get_area_list()])

                # culculate next state
                for vehicle in vehicle_manager.get_dispatched_vehicle_list():
                    area = area_manager.get_area_by_area_id(vehicle.location_area_id)
                    next_state = feature_manager.calc_state(area=area, demand_array=demand_array, supply_array=supply_array)
                    feature_manager.register_next_state(vehicle_id=vehicle.id, next_state_array=next_state)
                # =================================================================

                # ======================= calculate rewards =======================
                reward_culcurator.load(supply_array=supply_array, demand_array=demand_array)
                for vehicle in vehicle_manager.get_dispatched_vehicle_list():
                    from_area_id = feature_manager.get_from_area_id_by_vehicle_id(vehicle_id=vehicle.id)
                    to_area_id = feature_manager.get_to_area_id_by_vehicle_id(vehicle_id=vehicle.id)
                    reward = reward_culcurator.calc_reward(
                        start_area_id=from_area_id,
                        destination_area_id=to_area_id
                    )
                    action = feature_manager.get_action_by_vehicle_id(vehicle_id=vehicle.id)
                    feature_manager.register_reward(vehicle_id=vehicle.id, reward=reward)
                # =================================================================

                # ============================= learn =============================
                for vehicle_id, feature in feature_manager.get_whole_data():
                    dispatch_module.memorize(
                        state=feature["state"],
                        action=feature["action"],
                        next_state=feature["next_state"],
                        reward=feature["reward"],
                        from_area_id=feature["from_area_id"],
                        to_area_id=feature["to_area_id"],
                    )
                loss = dispatch_module.train(area_manager=area_manager, date_info=start_datetime_of_this_timeslice, episode=episode)
                # =================================================================

                # ===================== culculate next dispatch =====================
                prediction = demand_predictor.predict(
                    start_datetime=start_datetime_of_this_timeslice,
                    end_datetime=end_datetime_of_this_timeslice,
                    feature=None,
                    num_areas=area_manager.num_areas,
                    debug=config.DEBUG,
                )

                # plan dispatch order
                dispatch_order_list = dispatch_module(
                    area_manager=area_manager,
                    vehicle_manager=vehicle_manager,
                    prediction=prediction,
                    episode=episode,
                )

                # set dispatch order to simulator
                feature_manager.reset()
                for dispatch_order in dispatch_order_list:
                    feature_manager.register_action(vehicle_id=dispatch_order.vehicle_id, action=dispatch_order.action)
                    from_area_id = area_manager.node_id_to_area(node_id=dispatch_order.start_node_id).id
                    feature_manager.register_from_area_id(vehicle_id=dispatch_order.vehicle_id, from_area_id=from_area_id)
                    to_area_id = area_manager.node_id_to_area(node_id=dispatch_order.end_node_id).id
                    feature_manager.register_to_area_id(vehicle_id=dispatch_order.vehicle_id, to_area_id=to_area_id)
                simulator.set_dispatch_orders(dispatch_order_list)
                # =================================================================

                # proceed to next timeslice
                simulator.proceed()

            # finish simulation of the day
            # simulator.print_stats()
            # proceed to next day
            simulator.reload(date_module.date)
        checkpoint_path = dqn_checkpoint_dir / f"episode{episode+1}.cpt"
        dispatch_module.save(checkpoint_path)

        date_module = DataModule(demand_prediction_mode=DemandPredictionMode.TRAIN)
        simulator.reset(date_module.date)

    checkpoint_path = dqn_checkpoint_dir / f"dqn.cpt"
    dispatch_module.save(checkpoint_path)
