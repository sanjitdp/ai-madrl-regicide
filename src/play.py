from env import RegicideEnv
from agent import RegicideAgent
import numpy as np

# helper functions
def vectorize_obs(observation):
    print(type(observation))
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

def vectorize_action(action):
    action_vector = np.zeros(14)
    action_vector[0:7] = action[0]
    action_vector[7:]  = action[1]
    action_vector = [int(n) for n in action_vector]
    return action_vector

# hyperparameters
learning_rate = 0.01
n_episodes = 100000
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
env = RegicideEnv()
observation = vectorize_obs(env.reset())
game_over = False
turn_count = 0
reward = 0

# Start playing
env.render("start")
while not game_over:
    turn_count += 1
    action = env.action_space.sample()
    next_observation, game_over, reward = env.step(env.do_action(action))
    action, next_observation = vectorize_action(action), vectorize_obs(next_observation)
    agent.update(action, observation, game_over, reward, next_observation)
    if game_over:
        break
    else:
        env.render()
        observation = next_observation
print("Turn count:", turn_count)
agent.decay_epsilon(epsilon_decay)
