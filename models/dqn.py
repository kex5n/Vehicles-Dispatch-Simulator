from collections import namedtuple
import random
from typing import List

from objects import AreaManager

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

BATCH_SIZE = 64
CAPACITY = 500
GAMMA = 0.9

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "from_area_id", "to_area_id"))


class ReplayMemory:
    def __init__(self, capacity=500):
        self.__capacity = capacity
        self.__memory = []
        self.__index = 0

    def memorize(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, from_area_id: int, to_area_id: int) -> None:
        if len(self.__memory) < self.__capacity:
            self.__memory.append(None)
            self.__memory[self.__index] = Transition(state, action, next_state, reward, from_area_id, to_area_id)
            self.__index = (self.__index + 1) % self.__capacity

    def sample(self, batch_size) -> List[Transition]:
        return random.sample(self.__memory, batch_size)

    def __len__(self) -> int:
        return len(self.__memory)

class Q:
    def __init__(self, num_states: int, num_actions: int):
        self.num_actions = num_actions
        self.memory = ReplayMemory(capacity=CAPACITY)
        self.model = nn.Sequential()
        self.model.add_module("fc1", nn.Linear(num_states, 32))
        self.model.add_module("relu1", nn.ReLU())
        self.model.add_module("fc2", nn.Linear(32, 32))
        self.model.add_module("relu2", nn.ReLU())
        self.model.add_module("fc3", nn.Linear(32, num_actions))
        self.print = True

        print(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self, area_manager: AreaManager, date_info, episode=None):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.tensor(torch.from_numpy(np.array(batch.state)), dtype=torch.float32)
        action_batch = torch.from_numpy(np.array(batch.action)).reshape(-1,1)
        reward_batch = torch.from_numpy(np.array(batch.reward))
        next_states = torch.tensor(torch.from_numpy(np.array(batch.next_state)), dtype=torch.float32)
        from_area_id_batch = torch.from_numpy(np.array(batch.from_area_id))
        to_area_id_batch = torch.from_numpy(np.array(batch.to_area_id))
        self.model.eval()
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # select max next_state value from masked next_state values
        next_state_values = self.model(next_states)
        mask = np.array([[True for _ in range(self.num_actions)] for _ in range(len(next_states))])
        for i, to_area_id in enumerate(to_area_id_batch):
            to_area = area_manager.get_area_by_area_id(int(to_area_id))
            num_candidates = to_area.num_neighbors + 1
            mask[i, :num_candidates] = False
        next_state_values[mask] = -np.inf
        max_next_states = next_state_values.max(1)[0]
        expected_state_action_values = reward_batch + GAMMA * max_next_states
        self.model.train()
        loss = F.smooth_l1_loss(state_action_values.flatten(), expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def decide_action(self, state, episode, candidate_area_ids: List[int], is_train: bool = False):
        # action is the index of next_area_id in candidate_area_ids.
        reshape_state = state.reshape(1, -1)
        epsilone = 0.5 * (1 / (episode+1))
        if is_train:
            if (episode > 1) and (epsilone <= np.random.uniform(0, 1)):
            # if epsilone <= np.random.uniform(0, 1):
                self.model.eval()
                with torch.no_grad():
                    values = self.model(reshape_state)
                    mask = np.array([True for _ in range(self.num_actions)])
                    mask[:len(candidate_area_ids)] = False
                    values[0][mask] = -np.inf
                    action = torch.LongTensor([[values.max(1)[1].view(1, 1)]])
            else:
                if (episode < 1):
                # if False:
                    if (0.9 >= np.random.uniform(0, 1)):
                        action = torch.LongTensor([[
                            candidate_area_ids.index(4)
                        ]])
                    else:
                        action = torch.LongTensor([[
                            random.choice(
                                range(len(candidate_area_ids))
                            )
                        ]])
                else:
                    action = torch.LongTensor([[
                        random.choice(
                            range(len(candidate_area_ids))
                        )
                    ]])
        else:
            self.model.eval()
            with torch.no_grad():
                values = self.model(reshape_state)
                mask = np.array([True for _ in range(self.num_actions)])
                mask[:len(candidate_area_ids)] = False
                values[0][mask] = -np.inf
                action = torch.LongTensor([[values.max(1)[1].view(1, 1)]])
        action = int(action[0][0])
        return action

    def memorize(self, state, action, next_state, reward, from_area_id, to_area_id):
        self.memory.memorize(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            from_area_id=from_area_id,
            to_area_id=to_area_id,
        )

class DQN:
    def __init__(self, k: int, num_actions: int):
        self.Q = Q(num_states=3+k*2, num_actions=num_actions)

    def update_q_function(self, area_manager: AreaManager, date_info, episode=None):
        return self.Q.replay(area_manager=area_manager, date_info=date_info, episode=episode)

    def get_action(self, state, episode, candidate_area_ids: List[int], is_train=False):
        return self.Q.decide_action(state=state, episode=episode, candidate_area_ids=candidate_area_ids, is_train=is_train)

    def memorize(self, state, action, next_state, reward, from_area_id, to_area_id):
        self.Q.memorize(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            from_area_id=from_area_id,
            to_area_id=to_area_id,
        )

    def save_checkpoint(self, checkpoint_path: str) -> None:
        print(f"save checkpoint {checkpoint_path}")
        torch.save(
            {
                'model_state_dict': self.Q.model.state_dict(),
                'opt_state_dict': self.Q.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        cpt = torch.load(checkpoint_path)
        stdict_m = cpt['model_state_dict']
        stdict_o = cpt['opt_state_dict']
        self.Q.model.load_state_dict(stdict_m)
        self.Q.optimizer.load_state_dict(stdict_o)
