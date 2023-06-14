import numpy as np
from tqdm import tqdm

from games import CollaborativeModifiedRPS, ModifiedRockPaperScissors, PrisonersDilemma, StableRPS
from grid import Grid, GridVisualizer

if __name__ == "__main__":
    np.random.seed(0)
    # exp_name = "collaborative_rps"
    # grid = Grid(game=CollaborativeModifiedRPS)
    exp_name = "modified_rps"
    grid = Grid(game=ModifiedRockPaperScissors)
    # exp_name = "prisoners"
    # grid = Grid(game=PrisonersDilemma)
    # exp_name = "stable_rps"
    # grid = Grid(game=StableRPS)
    # exp_name = "stable_rps_1"
    # grid = Grid(game=StableRPS, p_agents=1)
    for i in tqdm(range(100)):
        grid.step()
        vis = GridVisualizer(grid)
        vis.visualize(i)
        vis.save(step=i, exp_name=exp_name)
    vis.save_payoff_plot(exp_name=exp_name)
