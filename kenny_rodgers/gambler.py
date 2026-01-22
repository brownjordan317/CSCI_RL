import random
import numpy as np

class Gambler:
    def __init__(self, verbose=True):
        self.initial = 20
        self.terminal_value = 90
        self.P = [0.5, 0.5]  # [Win probability, Lose probability]
        self.actions = [
            "all_in", 
            "target_bet", 
            "half_bet",
            "quarter_bet",
            "three_quarters_bet"
            ]
        self.curr_state = self.initial
        self.reward_total = 0
        self.verbose = verbose
        self.steps_taken = 0
        self.actions_taken = []

    def log(self, message):
        if self.verbose:
            print(f"  [Step {self.steps_taken}] {message}")

    def calc_reward(
            self, 
            curr_state, 
            og_state, 
            risk_taken
        ):
        # # Sparse reward: only given when reaching the goal
        # if curr_state >= self.terminal_value:
        #     reward = 1
        # elif self.curr_state <= 0:
        #     reward = -1
        # else:
        #     reward = 0
        # return reward

        reward = 0
        # big reward for reaching the goal
        if curr_state >= self.terminal_value:
            reward += 100
        # small reward for a win
        if curr_state - og_state > 0:
            reward += (1 - (1 * risk_taken))
        # small penalty for a loss
        if curr_state - og_state < 0:
            reward -= (1 - (1 * risk_taken))
        # big penalty for going bankrupt
        if curr_state <= 0:
            reward -= 100
        # small penalty for not reaching the goal
        # keeps from rewarding lots of turns
        reward -= 1
        return reward
        
    def place_bet(self, amount):
        flip = random.choices(
            ["Heads", "Tails"], 
            weights=self.P, 
            k=1)[0]
        win_amt = amount 
        
        if flip == "Heads":
            self.curr_state += win_amt
            self.log(f"Outcome: HEADS. Won ${win_amt}. New Balance: ${self.curr_state}")
        else:
            self.curr_state -= win_amt
            self.log(f"Outcome: TAILS. Lost ${win_amt}. New Balance: ${self.curr_state}")
        
        # Ensure state doesn't drop below 0
        self.curr_state = max(0, self.curr_state)

    def choose_action(self):
        action = random.choice(self.actions)
        return action
    
    def step(self, action):
        self.steps_taken += 1
        og_state = self.curr_state

        if action == "all_in":
            action = "all_in"
            bet_amount = self.curr_state
            self.log(f"Action: {action} | Bet: ${bet_amount}")
            self.place_bet(bet_amount)
        elif action == "target_bet":
            # Logic: Bet just enough to try to hit exactly 90 on a win
            bet_amount = int((self.terminal_value - self.curr_state))
            # Limit bet to current capital
            bet_amount = min(bet_amount, self.curr_state)
            self.log(f"Action: {action} | Bet: ${bet_amount}")
            self.place_bet(bet_amount)
        elif action == "half_bet":
            bet_amount = self.curr_state // 2
            self.log(f"Action: {action} | Bet: ${bet_amount}")
            self.place_bet(bet_amount)
        elif action == "quarter_bet":
            bet_amount = self.curr_state // 4
            self.log(f"Action: {action} | Bet: ${bet_amount}")
            self.place_bet(bet_amount)
        elif action == "three_quarters_bet":
            bet_amount = self.curr_state * 3 // 4
            self.log(f"Action: {action} | Bet: ${bet_amount}")
            self.place_bet(bet_amount)

        risk_taken = bet_amount / og_state
        self.actions_taken.append(action)
        reward = self.calc_reward(self.curr_state, og_state, risk_taken)
        self.reward_total += reward

best_reward = float('-inf')
num_episodes = 1000
total_successes = 0
best_actions = None

print(f"--- Starting Simulation: Goal ${90}, Starting ${20} ---\n")

for episode in range(1, num_episodes + 1):
    print(f"EPISODE {episode}:")
    robert = Gambler(verbose=True)
    terminated = False
    
    while not terminated:
        action = robert.choose_action()
        robert.step(action)
        
        # Termination conditions: Bankrupt or Goal Met
        if robert.curr_state >= robert.terminal_value:
            print(f"RESULT: SUCCESS! Final Capital: ${robert.curr_state}")
            terminated = True
            total_successes += 1
        elif robert.curr_state <= 0:
            print(f"RESULT: BUST. Final Capital: ${robert.curr_state}")
            terminated = True
            
    best_reward = max(best_reward, robert.reward_total)
    if best_reward == robert.reward_total:
        best_actions = robert.actions_taken
    print(f"Action taken: {robert.actions_taken}")
    print(f"Total Reward: {robert.reward_total}")
    print("-" * 30)

print(f"\nSimulation Complete. Max Reward Achieved: {best_reward}")
print(f"Best Actions: {best_actions}")
print(f"Total Succes Rate: {total_successes}/{num_episodes}")

