import random
import numpy as np

class Gambler:
    def __init__(self, verbose=True):
        self.initial = 20
        self.terminal_value = 90
        self.P = [0.8, 0.2]  # Win probability
        self.curr_state = self.initial
        self.reward_total = 0
        self.verbose = verbose
        self.steps_taken = 0
        self.actions_taken = []

    def log(self, message):
        if self.verbose:
            print(f"  [Step {self.steps_taken}] {message}")

    def calc_reward(self, curr_state):
        # Sparse reward: only given when reaching the goal
        return 1 if curr_state >= self.terminal_value else 0
        
    def place_bet(self, amount):
        flip = random.choices(["Heads", "Tails"], weights=self.P, k=1)[0]
        win_amt = amount 
        
        if flip == "Heads":
            self.curr_state += win_amt
            self.log(f"Outcome: HEADS. Won ${win_amt}. New Balance: ${self.curr_state}")
        else:
            self.curr_state -= win_amt
            self.log(f"Outcome: TAILS. Lost ${win_amt}. New Balance: ${self.curr_state}")
        
        # Ensure state doesn't drop below 0
        self.curr_state = max(0, self.curr_state)
    
    def step(self):
        self.steps_taken += 1
        prev_balance = self.curr_state

        if self.curr_state >= self.terminal_value:
            action = "STAY"
            self.log(f"Action: {action} (Goal reached)")
        elif self.terminal_value / self.curr_state > 2:
            action = "ALL-IN"
            bet_amount = self.curr_state
            self.log(f"Action: {action} | Bet: ${bet_amount}")
            self.place_bet(bet_amount)
        else:
            action = "PERCENTAGE_BET"
            # Logic: Bet just enough to try to hit exactly 90 on a win
            bet_amount = int((self.terminal_value - self.curr_state))
            # Limit bet to current capital
            bet_amount = min(bet_amount, self.curr_state)
            self.log(f"Action: {action} | Bet: ${bet_amount}")
            self.place_bet(bet_amount)

        self.actions_taken.append(action)
        reward = self.calc_reward(self.curr_state)
        self.reward_total += reward

# --- Execution ---
best_reward = float('-inf')
num_episodes = 100
total_successes = 0

print(f"--- Starting Simulation: Goal ${90}, Starting ${20} ---\n")

for episode in range(1, num_episodes + 1):
    print(f"EPISODE {episode}:")
    robert = Gambler(verbose=True)
    terminated = False
    
    while not terminated:
        robert.step()
        
        # Termination conditions: Bankrupt or Goal Met

        if robert.curr_state >= robert.terminal_value:
            print(f"RESULT: SUCCESS! Final Capital: ${robert.curr_state}")
            terminated = True
            total_successes += 1
        elif robert.curr_state <= 0:
            print(f"RESULT: BUST. Final Capital: ${robert.curr_state}")
            terminated = True
            
    best_reward = max(best_reward, robert.reward_total)
    print(f"Action taken: {robert.actions_taken}")
    print("-" * 30)

print(f"\nSimulation Complete. Max Reward Achieved: {best_reward}")
print(f"Total Succes Rate: {total_successes}/{num_episodes}")
