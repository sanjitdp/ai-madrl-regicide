import gymnasium as gym
from gymnasium.spaces import MultiBinary, Tuple, Dict, Discrete, MultiDiscrete
import numpy as np
import random

"""
Setup for a 2-player game of Regicide, to be played by a person through terminal or an AI model.

TODO:
* Fix incorrect card effects
* Fix duplicate cards being drawn
* Fix duplicate enemies appearing and not matching suits left
* Implement 2-player game
"""

class Card:
    def __init__(self, suit, number):
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
                "turn":             Discrete(2)
            }
        )
    # Helper functions ___________________________________________
    def print_rules(self):
        print("\n%% Rules %%%%%%") 
        print("◈  Each turn, you attack an enemy by playing one or more cards.")
        print("◈  If the enemy is still alive, you must suffer their attack by sacrificing cards.\n")
        print("◈  Cards have health and attack equal to their value.\n")
        print("◈  Animal companions have a value of one and can be paired with cards.")
        print("◈  Cards of the same value can be paired if their values sum to 10 or less.\n")
        print("◈  There are thRree sets of four enemies, and each set is stronger than the last.")
        print("◈  Dropping an enemy's health to exactly 0 is a perfect kill and it gets placed directly into the tavern.\n")
        print("◈  Each suit applies the following effects with x being the card value. Effects are negated if the enemy is of the same suit.")
        print("     ♥ hearts   — move x cards from discard pile to tavern")
        print("     ♦ diamonds — move x total cards from tavern to player and ally hands")
        print("     ♠ spades   — reduce enemy attack by x")
        print("     ♣ clubs    — double attack")

    def apply_suit(self, suits, attack):
        for suit in suits:
            if self.curr_enemy.suit != suit:

                if suit == "hearts":
                    selected = self.discard_cards[-attack:]
                    self.discard_cards = self.discard_cards[:-attack]
                    for s in selected:
                        print(s.name)
                    if selected:
                        [self.tavern_cards.insert(0,s) for s in selected[::-1]] # Move cards to bottom of tavern pile
                        for t in self.tavern_cards:
                            print(t.name)
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
                    print(f"Player:\t{old_P_len}/7 —> {len(self.player_cards)}/7")
                    print(f"Ally:\t{old_A_len}/7 —> {len(self.ally_cards)}/7")
                    

                if suit == "spades":
                    self.curr_enemy.attack = max(0, self.curr_enemy.attack - attack)

                if suit == "clubs":
                    self.curr_enemy.health -= attack

    def play_card(self, action):
        for a in action.split(','): # index error
            if len(a) > 1 or int(a) > len(self.player_cards):
                    print(f"Invalid index.")
                    return False, False

        if action == "R":   # print rules
            self.print_rules()
            print(f"\n%% Player {self.turn} turn %%%%%%")
            print("Play cards by inputting the index(es), comma-separated.")
            action = input("--> ")

        if len(action) == 1: # valid index(es), check plays
            play = int(action) - 1
            cards_played = [self.player_cards[play]]
        else:
            plays = []
            [plays.append(int(card)-1) for card in action.split(',')],
            cards_played = [self.player_cards[play] for play in plays]

        if len(cards_played) >= 2: # check card validity
            if any([card.attack != cards_played[0].attack for card in cards_played]) and len(cards_played) > 2: # Playing different-valued cards together
                print(f"Invalid play. Cards must have the same value, or paired with one animal companion, to be played together.\nAttempted to play: {[c.name for c in cards_played]}")
                return False, False
            if sum(card.attack for card in cards_played) > 10 and not any([card.attack == 1 for card in cards_played]): # Playing same-valued cards with sum > 10
                print(f"Invalid play. Combo card plays cannot sum to a value greater than 10.\nAttempted to play: {[c.name for c in cards_played]}")
                return False, False
            if len(cards_played) > 2 and any([card.attack == 1 for card in cards_played]): # Playing animal companions with more than one other card
                print(f"Invalid play. Animal companions can only be played with up to one additional card.\nAttempted to play: {[c.name for c in cards_played]}")
                return False, False
            if len(set(cards_played)) != len(cards_played): # Inputting same index 
                print(f"Invalid play. You cannot select the same card more than once per play: {[c.name for c in cards_played]}")
                return False, False
        
        print(f"You played {', '.join([c.name for c in cards_played])}")

        attack = 0
        suits = set()
        valid = True

        for card in cards_played:
            # print("c:", card.name)
            suits.add(card.suit)
            attack += card.attack
            # print("a:", attack.name)
            # Move card to discard pile
            self.player_cards.remove(card) # TODO implement or remove player_hand
            self.played_cards.append(card)
        
        # Suit(s) effect
        self.apply_suit(suits, attack) # Handles all suit effects

        enemy_is_dead = False
        self.curr_enemy.health -= attack

        # Enemy status
        if self.curr_enemy.health == 0:
            self.tavern_cards.append(self.curr_enemy)
            enemy_is_dead = True
            print("perfect kill!")
        elif self.curr_enemy.health < 0:
            self.discard_cards.append(self.curr_enemy)
            enemy_is_dead = True
        
        return enemy_is_dead, valid

    def swap_turn(self):
        self.turn = 1 if self.turn == 2 else 2

        # Swap player and ally hands
        temp_cards, temp_hand = self.player_cards, self.player_hand
        self.player_cards, self.player_hand = self.ally_cards, self.ally_hand
        self.ally_cards, self.ally_hand = temp_cards, temp_hand

    # Gym functions ___________________________________________
    def reset(self, num_players):
        super().reset()

        self.turn = 1 # 1 for player 1, 2 for player 2

        self.cards = []
        for suit in ['clubs', 'diamonds', 'hearts','spades']:
            self.cards.append(AnimalCompanion(suit))
            for number in np.arange(2,11): 
                self.cards.append(Card(suit, number))
        random.shuffle(self.cards)

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
        self.player_hand     = in_play[:max_hand]
        self.ally_hand       = in_play[max_hand:2*max_hand]
        self.player_cards    = self.tavern_cards[-max_hand:]
        self.ally_cards      = self.tavern_cards[-2*max_hand:-max_hand]
        self.played_cards    = []
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
            self.played_cards,
            self.discard_cards,
            self.tavern_cards] 

        info = {}

        return observation, info
 
    def step(self, action):
        print("step!", len(self.player_cards))
        game_over = False
        if len(self.player_cards) <= 0:
            print(f"\nNone of your champions remain standing... Surrounded, your ally's champions fall soon after.\n")
            print(f"Innocents perished as corruption overtook the kingdom.\n")
            print("☠\tGame over.\t☠")
            game_over = True
            return observation, game_over

        # Play turn
        enemy_is_dead, valid = self.play_card(action)
        
        if not valid: # Bot should be punished for invalid moves
            return observation, game_over
        
        if enemy_is_dead:
            print("\n—————————————————————") 
            print(f"\n⚔ {self.curr_enemy.name} defeated! ⚔")
            
            if len(self.curr_suits_left) == 0: # Moving up a rank (jacks -> queens -> kings)
                print(f"♛ Level defeated! ♛") 
                self.curr_level += 1
                if self.curr_level == 3:    # Game won
                    print(f"✦✦✦ —— ⚔ You've saved the kingdom from all corrupted regals! ⚔ —— ✦✦✦\n")
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
                print(f"The {self.curr_enemy.name} slaughtered your remaining champions...\n")
                print(f"Innocents perished as corruption overtook the kingdom.\n")
                print("☠\tGame over.\t☠")
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
                    if len(set(sacrificed_cards)) != len(sacrificed_cards):
                        print(f"You cannot select the same card twice.")
                    else:
                        for card in sacrificed_cards:
                            self.player_cards.remove(card)
                            self.played_cards.append(card)
                        break
        
        if len(self.ally_cards) <= 0:
            print(f"\nNone of your ally's champions remain standing... Surrounded, your champions fall soon after.\n")
            print(f"Innocents perished as corruption overtook the kingdom.\n")
            print("☠\tGame over.\t☠")
            game_over = True

        self.swap_turn()
        return observation, game_over  # return (observation, reward, terminated_bool (whether game is over), truncated=False (end game early), info)

    def render(self, turn="player"):
        if turn == "start":
            print("⚔ ——————— ♛  REGICIDE ♛ ——————— ⚔")
            print("The royals have been corrupted. Defeat all 12 to save the kingdom!")
            print("Press R for rules.")

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

""" Run """
env1 = RegicideEnv()
observation, info = env1.reset(num_players = 2)
game_over = False
env1.render("start")
while not game_over:
    observation, game_over = env1.step(input('--> '))
    # frame, reward, is_done = env1.step(env.action_space.sample())
    if game_over:
        break
    else:
        env1.render()
