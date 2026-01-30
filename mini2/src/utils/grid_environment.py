import numpy as np
from src.utils.grid_world import gridworld


class GridEnvironment:
    """
    Unified Grid Environment that works for:
    - P1: Windy Chasm (with curr_grid, terminal_map)
    - P2: Robot Motion Control (with curr_grid, terminal_map)
    - P3: Circuit Design (without curr_grid, uses direct MDP components)
    
    Usage for P1/P2:
    ----------------
    env = GridEnvironment(
        curr_grid=grid,
        terminal_map={...},
        action_map={...},
        gamma=0.99,
        step_penalty=-1,
        random_transition_type="windy",
        random_probability=0.5,
        start_state=(0, 3)
    )
    
    Usage for P3:
    -------------
    # First create MDP with gridworld
    S, A, P, r, terminals = gridworld(
        action_map=action_map,
        transitions_type="circuit",
        grid_size=3,
        step_penalty=-0.1,
        correct_reward=100,
        incorrect_penalty=-50
    )
    
    # Then create environment
    env = GridEnvironment(
        S=S,
        A=A,
        P=P,
        r=r,
        terminal_states=terminals,
        gamma=0.99
    )
    """
    
    def __init__(self, 
                 curr_grid=None,
                 terminal_map=None,
                 action_map=None,
                 gamma=0.99,
                 step_penalty=-1,
                 random_transition_type=None,
                 random_probability=0.0,
                 start_state=None,
                 # P3 direct parameters
                 S=None,
                 A=None,
                 P=None,
                 r=None,
                 terminal_states=None):
        
        # Check if this is P3 mode (direct MDP specification)
        if S is not None and A is not None and P is not None and r is not None:
            # P3 mode: Use provided MDP components directly
            self.S = S
            self.A = A
            self.P = P
            self.r = r
            self.terminal_states = terminal_states if terminal_states is not None else set()
            self.gamma = gamma
            
            # P3 doesn't have a visual grid
            self.curr_grid = None
            self.terminal_state_values = [-100, 100]  # Dummy values for compatibility
            self.grid_size = None
            self.terminal_map = None
            self.action_map = action_map
            self.random_transition_type = None
            self.random_probability = 0.0
            self.start_state = None
            self.current_state = None
            
            print(f"P3 Mode - Circuit Design")
            print(f"S size: {len(self.S)}")
            print(f"A: {self.A}")
            print(f"terminal_states: {len(self.terminal_states)}")
            
        else:
            # P1/P2 mode: Build MDP from grid
            self.curr_grid = curr_grid
            self.grid_size = self.curr_grid.shape
            self.terminal_map = terminal_map
            self.terminal_state_values = list(terminal_map.values())
            self.action_map = action_map
            self.gamma = gamma
            self.step_penalty = step_penalty
            self.random_transition_type = random_transition_type
            self.random_probability = random_probability
            
            # Initial Location
            self.start_state = start_state
            self.current_state = start_state

            # build MDP with spec-compliant wind model
            self.S, self.A, self.P, self.r, self.terminal_states = \
                gridworld(
                    self.curr_grid, 
                    terminal_map=self.terminal_map,
                    action_map=self.action_map,
                    transitions_type=self.random_transition_type,
                    random_probability=self.random_probability
            )
            
            print(f"P1/P2 Mode - Grid World")
            print(f"S size: {len(self.S)}")
            print(f"A: {self.A}")
            print(f"terminal_states: {self.terminal_states}")
    

    def reset(self):
        """
        Resets the agent to the initial location.

        Returns
        -------
        state : tuple
            The initial state.
        """
        if self.start_state is not None:
            self.current_state = self.start_state
            return self.current_state
        else:
            # P3 mode - no specific start state for reset
            return None

    def step(self, s, a):
        """
        Returns transitions with (probability, reward) tuples.
        
        Works for both P1/P2 and P3 modes.

        Parameters
        ----------
        s : tuple
            The current state.
        a : str
            The action taken.

        Returns
        -------
        transitions : dict
            A dictionary with the next states as keys and tuples of (probability, reward) as values.
        """
        transitions = {}
        
        # Check if state and action exist in P
        if s not in self.P or a not in self.P[s]:
            return transitions
        
        for s_next, prob in self.P[s][a].items():
            # Try to get reward from different possible formats
            # Format 1: r[(s, a, s_next)] (P1/P2 format)
            if isinstance(self.r, dict) and (s, a, s_next) in self.r:
                reward = self.r[(s, a, s_next)]
            # Format 2: r[s][a] (P3 format)
            elif isinstance(self.r, dict) and s in self.r and a in self.r[s]:
                reward = self.r[s][a]
            # Format 3: Default
            else:
                reward = 0.0
            
            transitions[s_next] = (prob, reward)
        
        return transitions