from env import RegicideEnv
from agent import RegicideAgent
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
n_episodes = 10000
start_epsilon = 1.0
epsilon_decay = start_epsilon / n_episodes / 10  # reduce the exploration over time
print("epsilon decay =", epsilon_decay)
final_epsilon = 0.1

# create agent
agent = RegicideAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# Create env
verbose = (n_episodes < 10)
env = RegicideEnv(verbose=verbose)
total_turns = 0 # running average of turn_count
total_eps = 0
turns_history = []

# Play and learn
for episode in tqdm(range(n_episodes)):
    observation = vectorize_obs(env.reset())
    game_over, turn_count, reward = False, 0, 0
    total_eps += 1

    while not game_over:
        turn_count += 1
        total_turns += 1
        
        # if verbose:
        #     print("turn:", turn_count)
        #     env.render()

        action = agent.get_action(observation)

        if not action: # no legal actions found
            game_over = True
            reward = -1
            # if verbose:
            #     print("No remaining legal actions. GAME OVER.")
            break

        next_observation, game_over, reward = env.step(env.do_action(action))

        action, next_observation = agent.ID_action(action), vectorize_obs(next_observation)
        agent.update(action, observation, game_over, reward, next_observation)
        
        observation = next_observation

    avg_turns = total_turns / total_eps
    turns_history.append(turn_count)
    # print(f"\nepisode {episode}  —  turn count: {turn_count}\t(avg: {str(avg_turns)[:6]})")
    agent.decay_epsilon(epsilon_decay)


print(f"final turn count: {turns_history[-1]}, avg turn count: {avg_turns}")
plt.plot(range(n_episodes), turns_history, c="indigo", lw=0.2)
plt.axhline(avg_turns, c="black", zorder=3, ls="--")
plt.xlabel("Episodes")
plt.ylabel("Game length (turns)")
plt.title("Game duration over episodes played")
plt.savefig("history")