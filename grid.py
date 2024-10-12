import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from games import GameRules, ModifiedRockPaperScissors, PrisonersDilemma


class Agent:
    def __init__(self, n_moves: int) -> None:
        self.strategy: int = np.random.choice([i + 1 for i in range(n_moves)])  # 1: rock, 2: paper, 3: scissors
        # self.strategy: int = 1
        self.just_changed_strategy: bool = False
        self.last_outcome: int = -1


class Grid:
    def __init__(self, size: int = 50, p_agents: float = 0.5, game: GameRules = PrisonersDilemma) -> None:
        self.size = size
        self.grid = np.full((size, size), fill_value=-1, dtype=np.int64)
        self.game: GameRules = game()
        self.n_agents: int = int(p_agents * size**2)
        self.parameters = {
            "m": 5,
            "r": 0.05,
            "q": 0.05,
        }

        positions: List[Tuple[int, int]] = self.assign_initial_positions()
        # positions: List[Tuple[int, int]] = self.square_initial_positions()
        self.agents: List[Agent] = [Agent(self.game.n_moves) for _ in range(self.n_agents)]
        for i, pos in enumerate(positions):
            self.grid[pos] = i

        self.total_payoffs: List[float] = []

    def assign_initial_positions(self) -> List[Tuple[int, int]]:
        positions: List[Tuple[int, int]] = []
        for _ in range(self.n_agents):
            while True:
                position: Tuple[int, int] = (np.random.randint(0, self.size), np.random.randint(0, self.size))
                if position not in positions:
                    positions.append(position)
                    break
        return positions

    def square_initial_positions(self) -> List[Tuple[int, int]]:
        positions: List[Tuple[int, int]] = []
        square_size = int(np.floor(np.sqrt(self.n_agents)))
        for i in range(square_size):
            for j in range(square_size):
                positions.append((i, j))
        rest = self.n_agents - square_size**2
        for i in range(rest):
            positions.append((square_size, i))

        return positions

    def get_neighbours(self, position: Tuple[int, int]) -> List[int]:
        neighbours: List[int] = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if abs(i + j) == 1:
                    neighbour_pos: Tuple[int, int] = ((position[0] + i) % self.size, (position[1] + j) % self.size)
                    if self.grid[neighbour_pos].item() != -1:
                        neighbours.append(self.grid[neighbour_pos].item())
        return neighbours

    def get_neighbour_strategies(self, neighbours: List[int]) -> List[int]:
        return [self.agents[neighbour].strategy for neighbour in neighbours]

    def get_agent_position(self, agent_id: int) -> Tuple[int, int]:
        try:
            i, j = np.nonzero(self.grid == agent_id)
            return (i.item(), j.item())
        except:
            print(agent_id)

    def calculate_outcome_position(self, position: Tuple[int, int], strategy: int) -> float:
        neighbours = self.get_neighbours(position)
        neighbour_strategies = self.get_neighbour_strategies(neighbours)
        outcomes = [
            self.game.calculate_outcome(strategy, neighbour_strategy) for neighbour_strategy in neighbour_strategies
        ]
        return sum(outcomes)

    def move_agent(self, old_position: Tuple[int, int], new_position: Tuple[int, int]) -> None:
        if old_position != new_position:
            agent_id = self.grid[old_position].item()
            self.grid[new_position] = agent_id
            self.grid[old_position] = -1

    def agent_step(self, agent_id: int) -> None:
        # Step 1: Compete against neighbours
        pos = self.get_agent_position(agent_id)
        total_outcome = self.calculate_outcome_position(pos, self.agents[agent_id].strategy)
        self.agents[agent_id].last_outcome = total_outcome

        # Step 2: Update strategy
        neighbours = self.get_neighbours(pos)
        neighbour_strategies = self.get_neighbour_strategies(neighbours)

        # Choose random strategy with probability r
        if np.random.rand() < self.parameters["r"]:
            self.agents[agent_id].just_changed_strategy = True
            if np.random.rand() < self.parameters["q"]:
                self.agents[agent_id].strategy = (
                    self.game.possible_strategies[0]  # cooperation
                    if np.random.rand() < self.parameters["q"]
                    else np.random.choice(self.game.possible_strategies[1:])  # defection chosen at random
                )

        # With probability 1 - r, imitate best neighbouring strategy
        else:
            if len(neighbours) > 0:
                neighbour_outcomes = [self.agents[neighbour].last_outcome for neighbour in neighbours]
                if max(neighbour_outcomes) > total_outcome:
                    self.agents[agent_id].just_changed_strategy = True
                    self.agents[agent_id].strategy = neighbour_strategies[np.argmax(neighbour_outcomes)]
                else:
                    self.agents[agent_id].just_changed_strategy = False

        # Step 3: Move to best empty cell
        # Neighbourhood is the (2m+1) x (2m+1) square around the agent
        best_outcome: float = total_outcome
        best_position: Tuple[int, int] = pos
        for i in range(-self.parameters["m"], self.parameters["m"] + 1):
            for j in range(-self.parameters["m"], self.parameters["m"] + 1):
                new_position = ((pos[0] + i) % self.size, (pos[1] + j) % self.size)
                if self.grid[new_position].item() == -1:
                    new_position_outcome = self.calculate_outcome_position(
                        new_position, self.agents[agent_id].strategy
                    )
                    if new_position_outcome > best_outcome:
                        best_outcome = new_position_outcome
                        best_position = new_position

        self.move_agent(pos, best_position)

    def step(self) -> None:
        # Take random permutation of agents
        agent_ids = np.random.permutation(self.n_agents)
        for agent_id in agent_ids:
            self.agent_step(agent_id)
        # Compute total step payoff
        self.total_payoff = sum([agent.last_outcome for agent in self.agents])
        self.total_payoffs.append(self.total_payoff)


class GridVisualizer:
    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        # Custom color for every strategy value
        unnormalized_colors = [
            (255, 255, 255),  # white
            (0, 48, 73),  # dark blue
            (214, 40, 40),  # dark red
            (247, 127, 0),  # orange
            (252, 191, 73),  # yellow
            (210, 149, 191),  # lilac
            (1, 0, 1),  # magenta
            (0, 1, 1),  # cyan
        ]
        self.colors = [(r / 255, g / 255, b / 255) for r, g, b in unnormalized_colors]
        assert len(self.colors) >= len(self.grid.game.possible_strategies), "Not enough colors for this game"

    def strategy_grid(self) -> np.ndarray:
        strat_grid = np.zeros_like(self.grid.grid)
        for i in range(self.grid.size):
            for j in range(self.grid.size):
                if self.grid.grid[i, j] != -1:
                    strat_grid[i, j] = self.grid.agents[self.grid.grid[i, j]].strategy
        return strat_grid

    def grid_to_rgb(self, strat_grid: np.ndarray) -> np.ndarray:
        rgb_grid = np.zeros((self.grid.size, self.grid.size, 3))
        for i in range(self.grid.size):
            for j in range(self.grid.size):
                rgb_grid[i, j] = self.colors[int(strat_grid[i, j])]
        return rgb_grid

    def generate_plot(self, step: int = None) -> None:
        strat_grid = self.strategy_grid()
        rgb_grid = self.grid_to_rgb(strat_grid)
        total_payoff = self.grid.total_payoff
        fig, ax = plt.subplots()
        ax.imshow(rgb_grid)
        ax.set_xticks([])
        ax.set_yticks([])
        if step is not None:
            ax.set_title(f"Step {step}")
            # ax.set_title(f"Step {step}\nTotal payoff: {total_payoff:.2f}")
        fig.tight_layout()
        return fig, ax

    def visualize(self, step: int = None) -> None:
        fig, ax = self.generate_plot(step)
        plt.show(block=False)
        plt.pause(0.01)
        plt.close()

    def save(self, step: int, exp_name: str) -> None:
        fig, ax = self.generate_plot(step)
        if not os.path.exists(f"plots/{exp_name}"):
            os.makedirs(f"plots/{exp_name}")
        plt.savefig(f"plots/{exp_name}/step_{step}.png", bbox_inches="tight", pad_inches=0.05)
        plt.close()

    def save_payoff_plot(self, exp_name: str) -> None:
        fig, ax = plt.subplots()
        ax.plot(self.grid.total_payoffs)
        ax.set_xlabel("Step")
        ax.set_ylabel("Total payoff")
        plt.savefig(f"plots/{exp_name}/payoff_plot.png")

        average_payoffs = [payoff / self.grid.n_agents for payoff in self.grid.total_payoffs]
        fig, ax = plt.subplots()
        ax.plot(average_payoffs)
        ax.set_xlabel("Step")
        ax.set_ylabel("Average payoff per agent")
        plt.savefig(f"plots/{exp_name}/avg_payoff_plot.png")
        plt.close()
