import numpy as np
from tqdm import tqdm

from games import ModifiedRockPaperScissors, PrisonersDilemma
from grid import Grid, GridVisualizer

if __name__ == "__main__":
    np.random.seed(0)
    grid = Grid(game=ModifiedRockPaperScissors)
    for i in tqdm(range(50)):
        grid.step()
        vis = GridVisualizer(grid)
        vis.visualize()
