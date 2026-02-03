import numpy as np
import random

class PipeOptions:
    def __init__(self):
        # Directions: N, E, S, W
        self.pipes = {
            # T pipes
            "T_0":   {"conn": {"N", "E", "W"}, "weight": 1},
            "T_90":  {"conn": {"N", "E", "S"}, "weight": 1},
            "T_180": {"conn": {"E", "S", "W"}, "weight": 1},
            "T_270": {"conn": {"N", "S", "W"}, "weight": 1},

            # Straight pipes
            "I_0":   {"conn": {"N", "S"}, "weight": 4},
            "I_90":  {"conn": {"E", "W"}, "weight": 4},

            # Corner pipes
            "L_0":   {"conn": {"N", "E"}, "weight": 3},
            "L_90":  {"conn": {"E", "S"}, "weight": 3},
            "L_180": {"conn": {"S", "W"}, "weight": 3},
            "L_270": {"conn": {"N", "W"}, "weight": 3},

            # 4-way cross
            "X_0":   {"conn": {"N", "E", "S", "W"}, "weight": 0.5},
        }

        self.opposite = {
            "N": "S", "S": "N",
            "E": "W", "W": "E"
        }

class PipeGrid:
    def __init__(self, rows, cols, pipe_options):
        self.rows = rows
        self.cols = cols
        self.opts = pipe_options
        self.grid = np.empty((rows, cols), dtype=object)
        self.generate()

    def generate(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r, c] = self.choose_valid_pipe(r, c)

    def choose_valid_pipe(self, r, c):
        required = set()

        # Check north neighbor
        if r > 0:
            north = self.grid[r - 1, c]
            if "S" in self.opts.pipes[north]["conn"]:
                required.add("N")

        # Check west neighbor
        if c > 0:
            west = self.grid[r, c - 1]
            if "E" in self.opts.pipes[west]["conn"]:
                required.add("W")

        candidates = []
        weights = []

        for pid, data in self.opts.pipes.items():
            if required.issubset(data["conn"]):
                candidates.append(pid)
                weights.append(data["weight"])

        return random.choices(candidates, weights=weights, k=1)[0]

class PipeVisualizerBW:
    """
    Convert the pipe grid into a black-and-white NumPy array.
    0 = black background
    1 = white path
    """
    def __init__(self, cell_size=5):
        self.cell_size = cell_size
        self.patterns = self.create_patterns(cell_size)

    def create_patterns(self, s):
        # s = cell size
        p = {}
        mid = s // 2

        # Initialize empty black square
        def empty():
            return np.zeros((s, s), dtype=int)

        # Straight vertical
        vert = empty()
        vert[:, mid] = 1
        p["I_0"] = vert

        # Straight horizontal
        hor = empty()
        hor[mid, :] = 1
        p["I_90"] = hor

        # Corners
        L0 = empty()
        L0[:mid+1, mid] = 1   # vertical up
        L0[mid, mid:] = 1     # horizontal right
        p["L_0"] = L0

        L90 = empty()
        L90[mid:, mid] = 1    # vertical down
        L90[mid, mid:] = 1    # horizontal right
        p["L_90"] = L90

        L180 = empty()
        L180[mid:, mid] = 1   # vertical down
        L180[mid, :mid+1] = 1 # horizontal left
        p["L_180"] = L180

        L270 = empty()
        L270[:mid+1, mid] = 1 # vertical up
        L270[mid, :mid+1] = 1 # horizontal left
        p["L_270"] = L270

        # T junctions
        T0 = empty()
        T0[mid, :] = 1        # horizontal
        T0[:mid+1, mid] = 1   # vertical up
        p["T_0"] = T0

        T90 = empty()
        T90[mid, mid:] = 1    # horizontal right
        T90[:, mid] = 1       # vertical
        p["T_90"] = T90

        T180 = empty()
        T180[mid, :] = 1      # horizontal
        T180[mid:, mid] = 1   # vertical down
        p["T_180"] = T180

        T270 = empty()
        T270[mid, :mid+1] = 1 # horizontal left
        T270[:, mid] = 1      # vertical
        p["T_270"] = T270

        # Cross
        X0 = empty()
        X0[mid, :] = 1
        X0[:, mid] = 1
        p["X_0"] = X0

        return p

    def render(self, grid):
        rows, cols = grid.shape
        s = self.cell_size
        canvas = np.zeros((rows * s, cols * s), dtype=int)

        for r in range(rows):
            for c in range(cols):
                canvas[r*s:(r+1)*s, c*s:(c+1)*s] = self.patterns[grid[r, c]]
        return canvas

def main():
    pipes = PipeOptions()
    grid = PipeGrid(10, 10, pipes)

    viz = PipeVisualizerBW(cell_size=5)
    bw_array = viz.render(grid.grid)
    import matplotlib.pyplot as plt

    # Assume bw_array is your black-and-white NumPy array
    plt.imshow(bw_array, cmap='gray', interpolation='nearest')
    plt.axis('off')  # Hide axes
    plt.savefig("pipe_grid.png", bbox_inches='tight', pad_inches=0)
    plt.show()       # Optional: display the image


    # Print as ASCII for visualization
    for row in bw_array:
        print("".join("█" if x else " " for x in row))

if __name__ == "__main__":
    main()
