import numpy as np

from games import ModifiedRockPaperScissors, PrisonersDilemma
from grid import Grid, GridVisualizer

if __name__ == "__main__":
    np.random.seed(0)
    grid = Grid(game=ModifiedRockPaperScissors)
    for i in range(50):
        print(f"Step {i}")
        grid.step()
        vis = GridVisualizer(grid)
        vis.visualize()
    print(vis.strategy_grid())
