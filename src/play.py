from env import RegicideEnv
from agent import RegicideAgent
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# TODO
# Bug spotted: "Invalid indices" for valid cards in player 2 attack turn.

# helper functions
def vectorize_obs(observation):
    obs_vector = np.zeros(24)
    obs_vector[0]     = observation["enemies_left"]
    obs_vector[1:4]   = observation["curr_suits_left"] + [0] * (3 - len(observation["curr_suits_left"])) # standardize length by adding 0's
    obs_vector[4]     = observation["enemy_suit"]
    obs_vector[5]     = observation["enemy_health"]
    obs_vector[6]     = observation["enemy_attack"]
    obs_vector[7]     = observation["num_discard"]
    obs_vector[8]     = observation["num_tavern"]
    obs_vector[9:16]  = observation["player_card_suits"] + [0] * (7 - len(observation["player_card_suits"]))
    obs_vector[16:23] = observation["player_card_values"] + [0] * (7 - len(observation["player_card_values"]))
    obs_vector[23 ]   = observation["num_ally_cards"]
    return tuple(obs_vector)

# hyperparameters
learning_rate = 0.001
n_episodes = 1
start_epsilon = 1.0
epsilon_decay = start_epsilon / n_episodes / 2  # reduce the exploration over time
final_epsilon = 0.1

# create agent
agent = RegicideAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# Create env
verbose = True
env = RegicideEnv(verbose=verbose)
total_turns = 0 # running average of turn_count
total_eps = 0
avg_turns_history = []

# Play and learn
for episode in tqdm(range(n_episodes)):
    observation = vectorize_obs(env.reset())
    game_over, turn_count, reward = False, 0, 0
    total_eps += 1

    while not game_over:
        turn_count += 1
        total_turns += 1
        
        if verbose:
            print("turn:", turn_count)
            env.render()

        action = agent.get_action(observation)

        if not action: # no legal actions found
            game_over = True
            reward = -1
            if verbose:
                print("No remaining legal actions. GAME OVER.")
            break

        next_observation, game_over, reward = env.step(env.do_action(action))

        action, next_observation = agent.ID_action(action), vectorize_obs(next_observation)
        agent.update(action, observation, game_over, reward, next_observation)
        
        observation = next_observation

    avg_turns = total_turns / total_eps
    avg_turns_history.append(avg_turns)
    print(f"\nepisode {episode}  —  turn count: {turn_count}\t(avg: {str(avg_turns)[:6]})")
    agent.decay_epsilon(epsilon_decay)


(f"final turn count: {avg_turns_history[-1]}")
plt.plot(avg_turns_history, range(n_episodes))
plt.savefig("history")