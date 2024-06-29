import gymnasium as gym
from gymnasium.spaces import MultiBinary, Tuple, Dict, Discrete, MultiDiscrete
import numpy as np
import random
from collections import defaultdict
from itertools import combinations

"""
AGENT
Multi-agent deep reinforcement learning model trained on 2-player Regicide using self-play
"""

# TODO 
# q val format: dict, obs's are keys, each corresponding value is an array of q-vals indexed by actions

class RegicideAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95, # for computing Q-value
    ):
        self.q_values = defaultdict(lambda: np.zeros(2**14))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def ID_action(self, action):
        # convert binary to unique number ID
        bin_action = [num for sub in action for num in sub] # Condense tuples into binary lists
        action_ID = int(''.join(map(str, bin_action)), 2)
        # print("ID_action:", action_ID)
        return action_ID

    def ID_to_action(self, action_ID):
        binary_str = bin(action_ID)[2:]  # Convert decimal ID to binary string without '0b' prefix
        padded_binary_str = binary_str.zfill(14)
        action = [int(bit) for bit in padded_binary_str]
        return action

    def get_legal_moves(self, obs_vector):
        card_vals = obs_vector[16:23]
        num_cards = np.count_nonzero(card_vals)
        """ for player attack turn """
        # ID special cards %%%%%%%%%%
        # ID animal companions
        animal_companion_idxs = []
        for i in range(len(card_vals)):
            if card_vals[i] == 1:
                animal_companion_idxs.append(i)

        # ID summables
        two_idxs, three_idxs, four_idxs, five_idxs = [], [], [], []
        nums = [2,3,4,5]
        idx_lists = [two_idxs, three_idxs, four_idxs, five_idxs]
        for num, idx_list in zip(nums, idx_lists):
            for i in range(len(card_vals)):
                if card_vals[i] == num:
                    idx_list.append(i)

        # Collect legal moves for attack turn %%%%%%%%%%
        legal_attacks = []

        # Yielding
        legal_attacks.append(np.zeros(7))

        # Selecting one card
        for i in range(num_cards):
            arr = np.zeros(7)
            arr[i] = 1
            legal_attacks.append(arr)
            # Selecting one card + animal companion
            for a in animal_companion_idxs:
                arr_with_animal_companion = np.copy(arr)
                arr_with_animal_companion[a] = 1
                legal_attacks.append(arr_with_animal_companion)

        # get all combo moves summing under 10
        for num, idx_list in zip(nums, idx_lists):
            combo_arr = np.zeros(7)
            for idx in idx_list:
                combo_arr = [1 if idx in idx_list else 0 for i in range(num_cards)]
            if sum(combo_arr) * num <= 10:
                legal_attacks.append(combo_arr)

        # Collect legal moves for defense turn %%%%%%%%%%
        damage = obs_vector[6]
        legal_sacrifices = []
        # get single card sacrifices
        for i in range(num_cards):
            if card_vals[i] >= damage:
                arr = np.zeros(7)
                arr[i] = 1
                legal_sacrifices.append(arr)
        
        # get multi-card sacrifices
        legal_sacrifices = []
        for r in range(num_cards):
            for indices in combinations(range(num_cards), r):
                if sum([card_vals[i] for i in indices]) >= damage:
                    combination = [1 if i in indices else 0 for i in range(num_cards)]
                    legal_sacrifices.append(combination)
        
        legal_moves = []
        for la in legal_attacks:
            la = np.pad(la, (0, max(0, 7 - len(la))), 'constant') # pad to length 7
            num_cards_left = num_cards - np.count_nonzero(la)
            for ls in legal_sacrifices:
                ls = np.pad(ls, (0, max(0, 7 - len(ls))), 'constant') # pad to length 7
                repeat_draw = False
                insufficient_cards = False
                for i in range(7):
                    if la[i] + ls[i] == 2: # if not both are 1
                        repeat_draw = True
                        break
                    if ls[i] == 1 and i >= num_cards_left - 1: # sacrifice indexes can't be larger than length of hand
                        insufficient_cards = True
                        break
                if sum([card_vals[i] for i in ls]) < damage:
                    insufficient_cards = True
                if not repeat_draw and not insufficient_cards:
                    legal_moves.append((la, ls))
        
        legal_moves = [ # This list contains duplicates
            (
                [int(num) for num in attacks],
                [int(num) for num in sacrifices]
            )
                for attacks, sacrifices in legal_moves]

        uniques = []
        [uniques.append(tup) for tup in legal_moves if tup not in uniques]

        if len(uniques) == 0: # if no legal moves (game over)
            # print("game about to be over. remaining move:", [(np.zeros(7).tolist(), np.pad(np.ones(num_cards), (0, max(0, 7 - num_cards)))).tolist()])
            return []
            # return [(np.zeros(7).tolist(), np.pad(np.ones(num_cards), (0, max(0, 7 - num_cards))).tolist())]

        return uniques
                    
    def get_action(self, observation):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        legal_moves = self.get_legal_moves(observation)

        # no remaining moves: game over
        if len(legal_moves) == 0:
            return None

        legal_move_IDs = [self.ID_action(move) for move in legal_moves] # ie [0, 1, ... 0] -> 12621

        # illegal moves have negative q values (indexing by move ID)
        for idx in range(2**14):
            if idx not in legal_move_IDs:
                self.q_values[observation][idx] = -999

        # with probability epsilon return a random LEGAL action to explore the environment
        if np.random.random() < self.epsilon:
            selected = random.sample(legal_moves, 1)[0]
            # print("selected randomly:", selected)
            return selected

        # with probability (1 - epsilon) act greedily
        else:  # should select only legal moves as illegal ones have neg. q-vals
            selected = self.ID_to_action(np.argmax(self.q_values[observation])) # np.argmax returns index of max value, which will be an action ID with highest q-val
            # print("Greedy selection:", selected)
            return selected

    def update(self, action, observation, game_over, reward, next_observation):
        # update action Q-value
        future_q_value = (not game_over) * np.max(self.q_values[next_observation])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[observation][action]
        )
        self.q_values[observation][action] = (
            self.q_values[observation][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self, epsilon_decay):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)
