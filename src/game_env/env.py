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
        royal_map = {11: "jack", 12: "queen", 13: "king"}
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
                    if selected:
                        self.tavern_cards.insert(0,selected) # Move cards to bottom of tavern pile
                    if selected in self.discard_cards:
                        self.discard_cards.remove(selected)
                if suit == "diamonds":
                    free_spaces = 14 - len(self.player_cards) - len(self.ally_cards) # NOTE 2-player only, generalize later
                    diamond_value = attack
                    while diamond_value and free_spaces and self.tavern_cards: # Each player draws until the card value is met, all hands are full, or tavern is empty
                        if 7 - len(self.player_cards):
                            self.player_cards.append(self.tavern_cards.pop())
                            free_spaces -= 1
                            diamond_value -= 1
                        if 7 - len(self.ally_cards):
                            self.ally_cards.append(self.tavern_cards.pop())
                            diamond_value -= 1
                if suit == "spades":
                    self.curr_enemy.attack = max(0, self.curr_enemy.attack - attack)
                if suit == "clubs":
                    self.curr_enemy.health -= attack

    def play_card(self, cards):
        attack = 0
        suits = set()

        if len(cards) != 1:
            animals = sum([c.attack == 1 for c in cards])
            for card in cards:
                suits.add(card.suit)
                attack += card.attack
                # Suit effect
                self.apply_suit(suits, attack)
                # Move card to discard pile
                self.player_cards.remove(card) # TODO implement or remove player_hand
                self.discard_cards.append(card)
        else:
            card = cards[0]
            attack = card.attack
            suits.add(card.suit)
            # Suit effect
            self.apply_suit(suits, attack)
            # Move card to discard pile
            self.player_cards.remove(card) # TODO implement or remove player_hand
            self.discard_cards.append(card)

        enemy_is_dead = False
        perfect_kill = False
        self.curr_enemy.health -= attack

        # Enemy status
        if self.curr_enemy.health == 0:
            self.tavern_cards.append(self.curr_enemy)
            enemy_is_dead = True
            perfect_kill = True
        elif self.curr_enemy.health < 0:
            self.discard_cards.append(self.curr_enemy)
            enemy_is_dead = True
            perfect_kill = False
        
        return enemy_is_dead, perfect_kill

    # Gym functions ___________________________________________
    def reset(self, num_players):
        super().reset()

        max_hand    = 9 - num_players
        in_play     = random.sample(range(0,40), num_players*max_hand) # TODO prevent duplicates
        random_suit = random.randint(0, 3) 
        suits_left  = ["hearts", "diamonds", "clubs", "spades"]
        suits_left.pop(random_suit)

        self.curr_level      = 0
        self.tavern_cards    = random.sample(cards, len(cards))
        self.enemies = [
        [EnemyCard('hearts', 11), EnemyCard('diamonds', 11), EnemyCard('clubs', 11), EnemyCard('spades', 11)],   # Jacks
        [EnemyCard('hearts', 12), EnemyCard('diamonds', 12), EnemyCard('clubs', 12), EnemyCard('spades', 12)],   # Queens
        [EnemyCard('hearts', 13), EnemyCard('diamonds', 13), EnemyCard('clubs', 13), EnemyCard('spades', 13)]]   # Kings
        self.curr_enemy      = self.enemies[self.curr_level][random_suit] # TODO avoid duplicates
        self.enemies_left    = 12
        self.curr_suits_left = suits_left
        self.enemy_attack    = 15 # enemies[curr_enemy].attack
        self.enemy_health    = 20 # enemies[curr_enemy].health
        self.num_discard     = 0
        self.num_tavern      = 26 # 52 - 12 enemies - 7*2 cards in both player hands
        self.player_hand     = in_play[:max_hand]
        self.ally_hand       = in_play[max_hand:2*max_hand]
        self.player_cards    = [self.tavern_cards[x] for x in self.player_hand]
        self.ally_cards      = [self.tavern_cards[x] for x in self.ally_hand]
        self.discard_cards   = []
        self.tavern_cards    = self.tavern_cards[num_players*max_hand:]

        game_running = True

        observation = [ # TODO remove unused ones
            self.curr_level,
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

        return observation, info

 
    def step(self, action):
        game_over = False
        if len(self.player_cards) == 0:
            print(f"None of your champions remain standing...\n")
            print(f"Innocents perished as corruption overtook the kingdom.\n")
            print("☠  Game over. ☠")
            game_over = True
            return observation, game_over

        if len(action) == 1:
            play = int(action) - 1
            print(f"You played: {self.player_cards[play].name}")
            enemy_is_dead, perfect_kill = self.play_card([self.player_cards[play]])

        else:
            plays = []
            [plays.append(int(card)-1) for card in action.split(',')],
            cards_played = [self.player_cards[play] for play in plays]
            if len(cards_played) >= 2: # card combo: combine same-numbered cards summing under 10
                if any([card.attack != cards_played[0].attack for card in cards_played]) and len(cards_played) > 2:
                    print(f"Invalid play. Cards must have the same value, or paired with one animal companion, to be played together.\nAttempted to play: {[c.name for c in cards_played]}")
                    return None
                if sum(card.attack for card in cards_played) > 10:
                    print(f"Invalid play. Combo card plays cannot sum to a value greater than 10.\nAttempted to play: {[c.name for c in cards_played]}")
                    return None
                if len(cards_played) > 2 and any([card.attack == 1 for card in cards_played]):
                    print(f"Invalid play. Animal companions can only be played with up to one additional card.\nAttempted to play: {[c.name for c in cards_played]}")
                    return None
            print(f"You played {', '.join([c.name for c in cards_played])}")
            enemy_is_dead, perfect_kill = self.play_card(cards_played)
        
        if enemy_is_dead:
            print(f"⚔ {self.curr_enemy.name} defeated! ⚔")
            
            if len(self.curr_suits_left) == 0: # Moving up a rank (jacks -> queens -> kings)
                print(f"♛ Level defeated! ♛") 
                self.curr_level += 1
                if self.curr_level == 3:    # Game won
                    print(f"⃰✦*✶⃰✶⃰✦✶⃰✶⃰٭✦ —— ⚔ You've saved the kingdom from all corrupted regals! ⚔ —— ✦٭✶⃰✶⃰✦✶⃰*✶⃰*\n")
                    game_over = True

            # Discard enemy card
            if perfect_kill:
                print("perfect kill!")
                self.tavern_cards.append(self.curr_enemy)
            else:
                self.discard_cards.append(self.curr_enemy)

            # Pull new enemy card            
            random_suit = random.sample(self.curr_suits_left)
            self.curr_suits_left.remove(random_suit)
            self.curr_enemy = self.enemies[self.curr_level][random_suit]

        else: # enemy attack turn
            self.render(turn="enemy") # see current enemy stats and cards in hand

            # no choice can win. game over!
            total_health = sum(card.health for card in self.player_cards)
            if total_health < self.curr_enemy.attack:
                print(f"The {self.curr_enemy} slaughtered your remaining champions...\n")
                print(f"Innocents perished as corruption overtook the kingdom.\n")
                print("☠ Game over. ☠")
                game_over = True
                return observation, game_over

            # player must select which cards to give up
            else:
                if self.curr_enemy.attack == 0:
                    return observation, game_over
                while True: # Loop if players choose invalid cards
                    sacrificed_indexes = input('--> ')
                    sacrificed = []
                    [sacrificed.append(int(i)-1) for i in sacrificed_indexes.split(',')]
                    sacrificed_cards = [self.player_cards[s] for s in sacrificed]
                    sacrificed_health = sum(card.health for card in sacrificed_cards)
                    
                    if sacrificed_health < self.curr_enemy.attack:
                        print(f"These cards do not suffice. They can only bear {sacrificed_health} damage.")
                    else:
                        for card in sacrificed_cards:
                            self.player_cards.remove(card)
                            self.discard_cards.append(card)
                        break

        return observation, game_over  # return (observation, reward, terminated_bool (whether game is over), truncated=False (end game early), info)

    def render(self, turn="player"):

        if turn == "player":
            print("\n%% Game stats %%%%%%")
            print("suits remaining:", ', '.join(self.curr_suits_left))
            print("discard:", len(self.discard_cards))
            print("tavern: ", len(self.tavern_cards))

        print("\n%% Current enemy %%%%%%")
        print("jack of", self.curr_enemy.suit)
        print("⚔:", self.curr_enemy.attack)
        print("♥:", self.curr_enemy.health)
        print("\n%% Your hand %%%%%%")
        [print(f"{i}) {c.name}") for i, c in zip(range(1, len(self.player_cards)+1), self.player_cards)]

        if turn == "player":
            print("\n%% Player turn %%%%%%")
            print("Play cards by inputting the index(es), comma-separated")

        if turn == "enemy":
            print(f"Select which cards to suffer {self.curr_enemy.attack} damage.")
        
        


""" Run """
env1 = RegicideEnv()
observation, info = env1.reset(2)
game_over = False
while not game_over:
    env1.render()
    observation, game_over = env1.step(input('--> '))
    # frame, reward, is_done = env1.step(env.action_space.sample())

    if game_over:
        break
