import gymnasium as gym
from gymnasium.spaces import MultiBinary, Tuple, Dict, Discrete, MultiDiscrete
import numpy as np
import random

"""
Setup for a 2-player game of Regicide, to be played by a person through terminal or an AI model.
"""


class Card:
    def __init__(self, suit, number):
        suits_map = {'clubs': 1, 'diamonds': 2, 'hearts': 3, 'spades': 4}
        # self.suit = suits_map[suit]
        self.suit = suit
        self.number = number

        if number not in {11, 12, 13}:
            self.attack = number
        elif number == 11:      # Jacks
            self.attack = 10
        elif number == 12:      # Queens
            self.attack = 15
        elif number == 13:      # Kings
            self.attack = 20
        
        self.health = self.attack
        self.name = f"{number} of {self.suit}"

class EnemyCard(Card):
    def __init__(self, suit, number):
        super().__init__(suit, number)

        self.health = self.attack * 2
        royal_map = {11: "jacks", 12: "queens", 13: "kings"}
        self.name = f"{royal_map[number]} of {self.suit}"

class AnimalCompanion(Card):
    def __init__(self, suit, number=1):
        super().__init__(suit, 1)
        self.name = f"animal companion (A) of {self.suit}"

cards = []
for suit in ['clubs', 'diamonds', 'hearts','spades']:
    cards.append(AnimalCompanion(suit))
    for number in np.arange(2,11): # Include 0-value cards if jesters are in play
        cards.append(Card(suit, number))

def play_card(card):
    print("!")

class RegicideEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = Tuple([MultiBinary(7), MultiBinary(7)])
        self.observation_space = Dict(
            {
                "curr_enemy":       Discrete(12),
                "enemies_left":     Discrete(12),
                "curr_suits_left":  MultiDiscrete([5] * 3),
                "enemy_health":     Discrete(21),
                "enemy_attack":     Discrete(21),
                "num_discard":      Discrete(53),
                "num_tavern":       Discrete(53),
                "player_hand":      MultiDiscrete([53] * 7),
                "ally_hand":        MultiDiscrete([53] * 7),
            }
        )

    # Helper functions ___________________________________________
    def get_card_ID(card):
        return (card.number - 1) * 4 + card.suit

    def apply_suit(self, suits, attack):
        for suit in suits:
            if self.curr_enemy.suit != suit:
                if suit == "hearts":
                    selected = random.sample(self.discard_cards, min(len(self.discard_cards), attack))
                    print("selected for hearts: ", selected)
                    self.tavern_cards.append(selected)
                    self.discard_cards.remove(selected)
                if suit == "diamonds":
                    pass # TODO
                if suit == "spades":
                    self.curr_enemy.attack -= attack
                if suit == "clubs":
                    self.curr_enemy.health -= attack

    def play_card(self, cards): # TODO implement pairs of cards
        attack = 0
        suits = set()

        
        if len(cards) != 1:
            print("!")
            animals = sum([c.attack == 1 for c in cards])
            for card in cards:
                suits.add(card.suit)
                attack += card.attack
                self.player_cards.remove(card) # TODO implement or remove player_hand
                self.discard_cards.append(card)
        else:
            card = cards[0]
            print(card)
            attack = card.attack
            suits.add(card.suit)
            self.player_cards.remove(card) # TODO implement or remove player_hand
            self.discard_cards.append(card)

        enemy_is_dead = False
        self.curr_enemy.health -= attack

        # Suit effect
        self.apply_suit(suits, attack)

        # Enemy status
        if self.curr_enemy.health == 0:
            self.tavern_cards.append(curr_enemy)
            enemy_is_dead = True
        elif self.curr_enemy.health < 0:
            self.discard_cards.append(curr_enemy)
            enemy_is_dead = True
        
        return enemy_is_dead

    # Gym functions ___________________________________________
    def reset(self, num_players):
        super().reset()

        max_hand        = 9 - num_players
        in_play         = random.sample(range(0,40), num_players*max_hand) # TODO prevent duplicates
        
        self.enemies = [
        [EnemyCard('hearts', 11), EnemyCard('diamonds', 11), EnemyCard('clubs', 11), EnemyCard('spades', 11)],   # Jacks
        [EnemyCard('hearts', 12), EnemyCard('diamonds', 12), EnemyCard('clubs', 12), EnemyCard('spades', 12)],   # Queens
        [EnemyCard('hearts', 13), EnemyCard('diamonds', 13), EnemyCard('clubs', 13), EnemyCard('spades', 13)]]   # Kings
        self.curr_enemy      = self.enemies[0][random.randint(0, 3)] # TODO avoid duplicates
        self.enemies_left    = 12
        self.curr_suits_left = 3
        self.enemy_attack    = 15 # enemies[curr_enemy].attack
        self.enemy_health    = 20 # enemies[curr_enemy].health
        self.num_discard     = 0
        self.num_tavern      = 26 # 52 - 12 enemies - 7*2 cards in both player hands
        self.player_hand     = in_play[:max_hand]
        self.ally_hand       = in_play[max_hand:2*max_hand]
        self.player_cards    = [cards[x] for x in self.player_hand]
        self.ally_cards      = [cards[x] for x in self.ally_hand]
        self.discard_cards   = []
        self.tavern_cards    = []

        game_running = True

        observation = [
            self.enemies,
            self.curr_enemy,   
            self.enemies_left,   
            self.curr_suits_left,
            self.enemy_attack,   
            self.enemy_health,   
            self.num_discard,    
            self.num_tavern,     
            self.player_hand,    
            self.ally_hand,
            self.player_cards,
            self.ally_cards,
            self.discard_cards,
            self.tavern_cards] 

        info = {} # TODO

        return (observation, info)

    def step(self, action):

        if len(action) == 1:
            play = int(action) - 1
            print(f"You played: {self.player_cards[play].name}")
            self.play_card([self.player_cards[play]])

        else:
            # TODO implement rules (can only add aces or same-numbered cards summing under 10)
            plays = [int(card)-1 for card in action.split(',')]
            cards_played = [self.player_cards[play] for play in plays]
            print(f"You played {[c.name for c in cards_played]}")
            self.play_card(cards_played)

        

        return None
        # return (observation, reward, terminated_bool (whether game is over), truncated=False (end game early), info)

    def render(self):
        print("\n%% Current enemy %%%%%%")
        print("jack of", self.curr_enemy.suit)
        print("⚔:", self.curr_enemy.attack)
        print("♥:", self.curr_enemy.health)
        print("\n%% Your hand %%%%%%")
        [print(f"{i})  {c.name}") for i, c in zip(range(1, len(self.player_cards)+1), self.player_cards)]

        print("\n%% Player turn %%%%%%")
        print("play cards by inputting the index/es, comma-separated")

        


""" Test """
env1 = RegicideEnv()
observation, info = env1.reset(2)
done = False
while not done:
    env1.render()
    play = input('--> ')
    env1.step(play)
    # frame, reward, is_done = env1.step(env.action_space.sample())
    env1.render()
    done = True # temp

    if done:
        break
