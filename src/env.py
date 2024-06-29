import gymnasium as gym
from gymnasium.spaces import MultiBinary, Tuple, Dict, Discrete, MultiDiscrete
import numpy as np
import random

"""
REGICIDE ENV
Setup for a 2-player game of Regicide, to be played by a person through terminal or an AI model.

%%%%%%%%

* Action space
Draw one or more valid cards by index (format: #,#,#).

* Observation space
Enemies left:           number of enemies left, list of current suits left, 
Current enemy stats:    enemy suit, enemy health, enemy attack,
Cards to draw:          number of cards in discard pile, number of cards in tavern deck,
Cards in hand:          list of cards in hand

* Rewards
Defeat jack     + 1
Defeat queen    + 2
Defeat king     + 3
Lose game       - 1
Invalid move    - 999999
"""

class Card:
    def __init__(self, suit, number):
        self.suit = suit
        self.number = number
        suit_map = {'hearts': 0, 'diamonds': 1, 'spades': 2, 'clubs': 3}
        self.ID = suit_map[self.suit] * 13 + self.number

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

class RegicideEnv(gym.Env):
    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.suit_map = {'hearts': 1, 'diamonds': 2, 'spades': 3, 'clubs': 4}
        self.action_space = Tuple([MultiBinary(7), MultiBinary(7),])
        self.observation_space = Dict(
            {
                # Enemies left
                "enemies_left":     Discrete(12),
                "curr_suits_left":  MultiDiscrete([5] * 3),

                # Current enemy stats
                "enemy_suit":       Discrete(4),
                "enemy_health":     Discrete(21),
                "enemy_attack":     Discrete(41),

                # Cards to draw
                "num_discard":      Discrete(53),
                "num_tavern":       Discrete(53),

                # Cards in hand
                "player_card_suits":    MultiDiscrete([5] * 7), 
                "player_card_values":   MultiDiscrete([14] * 7),
            }
        )

    # Helper functions ___________________________________________
    def do_action(self, action):
        player_attack = [i + 1 for i, x in enumerate(action[0]) if x == 1]
        player_defend = [i + 1 for i, x in enumerate(action[1]) if x == 1]
        return (player_attack, player_defend)

    def apply_suit(self, suits, attack):
        for suit in suits:
            if self.curr_enemy.suit != suit:

                if suit == "hearts":
                    selected = self.discard_cards[-attack:]
                    self.discard_cards = self.discard_cards[:-attack]
                    if selected:
                        [self.tavern_cards.insert(0,s) for s in selected[::-1]] # Move cards to bottom of tavern pile
                    if selected in self.discard_cards:
                        self.discard_cards.remove(selected) # Remove moved cards from discard

                if suit == "diamonds":
                    old_P_len, old_A_len = len(self.player_cards), len(self.ally_cards)
                    diamond_value = attack
                    while diamond_value and ((len(self.player_cards) + len(self.ally_cards)) < 14) and self.tavern_cards: # Each player draws until the card value is met, all hands are full, or tavern is empty
                        if len(self.player_cards) < 7:
                            self.player_cards.append(self.tavern_cards.pop())
                            diamond_value -= 1
                        if len(self.ally_cards) < 7:
                            self.ally_cards.append(self.tavern_cards.pop())
                            diamond_value -= 1
                    print(f"Player:\t{old_P_len}/7 —> {len(self.player_cards)}/7") if self.verbose else None
                    print(f"Ally:\t{old_A_len}/7 —> {len(self.ally_cards)}/7") if self.verbose else None
                    
                if suit == "spades":
                    self.curr_enemy.attack = max(0, self.curr_enemy.attack - attack)

                if suit == "clubs":
                    self.curr_enemy.health -= attack

    def play_card(self, action):
        if not action:
            print("Yielding.") if self.verbose else None
            return False, True

        if len(action) > 1 or int(action[0]) > len(self.player_cards):
                print(f"Invalid index. Selected indices: {action}") if self.verbose else None
                return False, False

        if len(action) == 1: # valid index(es), check plays
            play = int(action[0]) - 1
            cards_played = [self.player_cards[play]]
        else:
            plays = []
            [plays.append(int(card)-1) for card in action.split(',')],
            cards_played = [self.player_cards[play] for play in plays]

        if len(cards_played) >= 2: # check card validity
            if any([card.attack != cards_played[0].attack for card in cards_played]) and len(cards_played) > 2: # Playing different-valued cards together
                print(f"Invalid play. Cards must have the same value, or paired with one animal companion, to be played together.\nAttempted to play: {', '.join([c.name for c in cards_played])}") if self.verbose else None
                return False, False
            if sum(card.attack for card in cards_played) > 10 and not any([card.attack == 1 for card in cards_played]): # Playing same-valued cards with sum > 10
                print(f"Invalid play. Combo card plays cannot sum to a value greater than 10.\nAttempted to play: {', '.join([c.name for c in cards_played])}") if self.verbose else None
                return False, False
            if len(cards_played) > 2 and any([card.attack == 1 for card in cards_played]): # Playing animal companions with more than one other card
                print(f"Invalid play. Animal companions can only be played with up to one additional card.\nAttempted to play: {', '.join([c.name for c in cards_played])}") if self.verbose else None
                return False, False
            if len(set(cards_played)) != len(cards_played): # Inputting same index 
                print(f"Invalid play. You cannot select the same card more than once per play: {', '.join([c.name for c in cards_played])}") if self.verbose else None
                return False, False
        
        print(f"You played {', '.join([c.name for c in cards_played])}") if self.verbose else None

        attack = 0
        suits = set()
        valid = True

        for card in cards_played:
            suits.add(card.suit)
            attack += card.attack
            # Move card to discard pile
            self.player_cards.remove(card)
            self.played_cards.append(card)
        
        # Suit(s) effect
        self.apply_suit(suits, attack) # Handles all suit effects

        enemy_is_dead = False
        self.curr_enemy.health -= attack

        # Enemy status
        if self.curr_enemy.health == 0:
            self.tavern_cards.append(self.curr_enemy)
            enemy_is_dead = True
            print("perfect kill!") if self.verbose else None
        elif self.curr_enemy.health < 0:
            self.discard_cards.append(self.curr_enemy)
            enemy_is_dead = True
        
        return enemy_is_dead, valid

    def sacrifice_card(self, action):
        if len(action) == 0 and self.enemy_attack > 0:
            print("No cards selected.") if self.verbose else None
            return False

        sacrificed_indexes = action

        for sacrificed_index in sacrificed_indexes:
            if int(sacrificed_index) > len(self.player_cards) or int(sacrificed_index) <= 0:
                print(f"Invalid index. Selected indices: {sacrificed_indexes}") if self.verbose else None
                return False

        sacrificed_cards = [self.player_cards[int(i)-1] for i in sacrificed_indexes]
        sacrificed_health = sum(card.health for card in sacrificed_cards)
        print(f"You selected {', '.join([card.name for card in sacrificed_cards])} for sacrifice.") if self.verbose else None
        

        if sacrificed_health < self.curr_enemy.attack:
            print(f"These cards do not suffice. They can only bear {sacrificed_health} damage.") if self.verbose else None
            return False
        if len(set(sacrificed_cards)) != len(sacrificed_cards):
            print(f"You cannot select the same card twice.") if self.verbose else None
            return False

        for card in sacrificed_cards:
            self.player_cards.remove(card)
            self.discard_cards.append(card)
    
        return True

    def swap_turn(self):
        self.turn = 1 if self.turn == 2 else 2

        # Swap player and ally hands
        temp_cards = self.player_cards
        self.player_cards = self.ally_cards
        self.ally_cards = temp_cards

    # Gym functions ___________________________________________
    def reset(self):
        super().reset()

        self.turn = 1 # 1 for player 1, 2 for player 2

        self.cards = []
        for suit in ['clubs', 'diamonds', 'hearts','spades']:
            self.cards.append(AnimalCompanion(suit))
            for number in np.arange(2,11): 
                self.cards.append(Card(suit, number))
        random.shuffle(self.cards)

        num_players = 2
        max_hand    = 9 - num_players
        in_play     = self.cards[-num_players*max_hand:]
        self.cards  = self.cards[:-num_players*max_hand] # remove cards in play
        random_suit = random.randint(0, 3) 
        suits_left  = ["hearts", "diamonds", "clubs", "spades"]
        suits_left.pop(random_suit)

        self.curr_level      = 0
        self.tavern_cards    = self.cards
        self.enemies = [
        [EnemyCard('hearts', 11), EnemyCard('diamonds', 11), EnemyCard('clubs', 11), EnemyCard('spades', 11)],   # Jacks
        [EnemyCard('hearts', 12), EnemyCard('diamonds', 12), EnemyCard('clubs', 12), EnemyCard('spades', 12)],   # Queens
        [EnemyCard('hearts', 13), EnemyCard('diamonds', 13), EnemyCard('clubs', 13), EnemyCard('spades', 13)]]   # Kings
        self.curr_enemy      = self.enemies[self.curr_level][random_suit] # TODO avoid duplicates
        del self.enemies[self.curr_level][random_suit]
        self.enemies_left    = 12
        self.curr_suits_left = suits_left
        self.enemy_attack    = 15 # enemies[curr_enemy].attack
        self.enemy_health    = 20 # enemies[curr_enemy].health
        self.num_discard     = 0
        self.num_tavern      = 26 # 52 - 12 enemies - 7*2 cards in both player hands
        self.player_cards    = in_play[:max_hand]
        self.ally_cards      = in_play[max_hand:2*max_hand]
        self.played_cards    = []
        self.discard_cards   = []
        self.tavern_cards    = self.cards

        self.obs = {
            # Enemies left
            "enemies_left":     12,
            "curr_suits_left":  [self.suit_map[s] for s in self.curr_suits_left],

            # Current enemy stats
            "enemy_suit":       random_suit,
            "enemy_health":     self.enemy_health,
            "enemy_attack":     self.enemy_attack,

            # Cards to draw
            "num_discard":      self.num_discard,
            "num_tavern":       self.num_tavern,

            # Cards in hand
            "player_card_suits":    [self.suit_map[c.suit] for c in self.player_cards],
            "player_card_values":   [c.attack for c in self.player_cards],
            "num_ally_cards":       len(self.ally_cards)
            }

        return self.obs
 
    def step(self, action):
        game_over = False
        reward = 0

        if len(self.player_cards) <= 0:
            print(f"\nNone of your champions remain standing... Surrounded, your ally's champions fall soon after.\n") if self.verbose else None
            print(f"Innocents perished as corruption overtook the kingdom.\n") if self.verbose else None
            print("☠\tGame over.\t☠") if self.verbose else None
            game_over = True
            reward = -1
            return self.obs, game_over, reward

        # Play turn
        enemy_is_dead, valid = self.play_card(action[0])
        
        if not valid: # Bot should be punished for invalid moves
            game_over = True
            reward = -999999
            return self.obs, game_over, reward
        
        if enemy_is_dead:
            print("\n—————————————————————")  if self.verbose else None
            print(f"\n⚔\t{self.curr_enemy.name} defeated!\t⚔") if self.verbose else None
            reward = self.curr_level + 1
            
            if len(self.curr_suits_left) == 0: # Moving up a rank (jacks -> queens -> kings)
                print(f"♛\tlevel defeated!\t♛")  if self.verbose else None
                self.curr_level += 1
                if self.curr_level == 3:    # Game won
                    print(f"✦✦✦ —— ⚔ You've saved the kingdom from all corrupted regals! ⚔ —— ✦✦✦\n") if self.verbose else None
                    game_over = True
                else:
                    self.curr_suits_left = ["hearts", "diamonds", "clubs", "spades"]

            # Discard played cards
            [self.discard_cards.append(c) for c in self.played_cards]
            self.played_cards = []

            # Pull new enemy card      
            random_suit = random.randint(0, len(self.curr_suits_left)-1) 
            self.curr_suits_left.pop(random_suit)
            self.curr_enemy = self.enemies[self.curr_level][random_suit]
            del self.enemies[self.curr_level][random_suit]

        else: # enemy attack turn
            self.render(turn="enemy") # see current enemy stats and cards in hand

            # no choice can win. game over!
            total_health = sum(card.health for card in self.player_cards)
            if total_health < self.curr_enemy.attack:
                print(f"The {self.curr_enemy.name} slaughtered your remaining champions... Surrounded, your ally's champions fell soon after.\n") if self.verbose else None
                print(f"Innocents perished as corruption overtook the kingdom.\n") if self.verbose else None
                print("☠\tGame over.\t☠") if self.verbose else None
                game_over = True
                reward = -1
                return self.obs, game_over, reward

            # player must select which cards to give up
            else:
                if self.curr_enemy.attack == 0:
                    return self.obs, game_over, reward

                valid = self.sacrifice_card(action[1]) # Returns whether selection is valid and handles card transfer

                if not valid:
                    game_over = True
                    reward = -999999
        
        if len(self.ally_cards) <= 0:
            print(f"\nNone of your ally's champions remain standing... Surrounded, your champions fall soon after.\n") if self.verbose else None
            print(f"Innocents perished as corruption overtook the kingdom.\n") if self.verbose else None
            print("☠\tGame over.\t☠") if self.verbose else None
            game_over = True
            reward = -1

        self.obs = {
            # Enemies left
            "enemies_left":     self.enemies_left,
            "curr_suits_left":  [self.suit_map[s] for s in self.curr_suits_left],

            # Current enemy stats
            "enemy_suit":       self.suit_map[self.curr_enemy.suit],
            "enemy_health":     self.enemy_health,
            "enemy_attack":     self.enemy_attack,

            # Cards to draw
            "num_discard":      self.num_discard,
            "num_tavern":       self.num_tavern,

            # Cards in hand
            "player_card_suits":    [self.suit_map[c.suit] for c in self.player_cards],
            "player_card_values":   [c.attack for c in self.player_cards],
            "num_ally_cards":       len(self.ally_cards),
        }

        self.swap_turn()
        return self.obs, game_over, reward

    def render(self, turn="player"):
        if not self.verbose:
            return None
            
        if turn == "start":
            print("⚔ ——————— ♛  REGICIDE ♛ ——————— ⚔")
            print("The royals have been corrupted. Defeat all 12 to save the kingdom!")

            turn = "player"

        if turn == "player":
            print("\n—————————————————————")  
            print("\n%% Game stats %%%%%%")
            print("suits remaining:", ', '.join(self.curr_suits_left))
            print("discard:", len(self.discard_cards))
            print("tavern: ", len(self.tavern_cards))
            print(f"your hand: {len(self.player_cards)}/7")
            print(f"ally hand: {len(self.ally_cards)}/7")

        if self.played_cards:
            print("\n%% Play area %%%%%%")
            print(', '.join([c.name for c in self.played_cards]))

        print("\n%% Current enemy %%%%%%")
        print(self.curr_enemy.name)
        print("♥:", self.curr_enemy.health)
        print("⚔:", self.curr_enemy.attack)
        print("\n%% Your hand %%%%%%")
        [print(f"{i}) {c.name}") for i, c in zip(range(1, len(self.player_cards)+1), self.player_cards)]

        if turn == "player":
            print(f"\n%% Player {self.turn} turn %%%%%%")
            print("Play cards by inputting the index(es), comma-separated.")

        if turn == "enemy":
            if self.curr_enemy.attack:
                print(f"Player {self.turn}: Select which cards to suffer {self.curr_enemy.attack} damage.")

# """ Run """
# env1 = RegicideEnv()
# observation = env1.reset()
# game_over = False
# env1.render("start")
# turn_count = 0
# reward = 0

# while not game_over:
#     turn_count += 1
#     observation, game_over, reward = env1.step(env1.do_action(env1.action_space.sample()))
#     if game_over:
#         break
#     else:
#         env1.render()

# print("Turn count:", turn_count)
