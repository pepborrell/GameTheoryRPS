from typing import List


class GameRules:
    def __init__(self) -> None:
        self.possible_strategies = []
        pass

    def calculate_outcome(self, strategy_1: int, strategy_2: int) -> int:
        """
        Returns the outcome for player 1 of a rock paper scissors game
        """
        raise NotImplementedError

    @property
    def n_moves(self) -> int:
        return len(self.possible_strategies)


class PrisonersDilemma(GameRules):
    def __init__(self) -> None:
        super().__init__()
        self.T = 1.3
        self.R = 1.0
        self.P = 0.1
        self.S = 0.0

        self.possible_strategies: List[int] = [1, 2]

    def is_strategy_valid(self, strategy: int) -> bool:
        return strategy in self.possible_strategies

    def get_move(self, strategy: int) -> str:
        assert self.is_strategy_valid(strategy)

        if strategy == 1:
            return "C"
        if strategy == 2:
            return "D"

    def calculate_outcome(self, strategy_1: int, strategy_2: int) -> int:
        # Transform strategy to move in ['C', 'D'] for readability
        move_1 = self.get_move(strategy_1)
        move_2 = self.get_move(strategy_2)

        if move_1 == "C":
            if move_2 == "C":
                return self.R
            if move_2 == "D":
                return self.S
        if move_1 == "D":
            if move_2 == "C":
                return self.T
            if move_2 == "D":
                return self.P

        raise ValueError("Invalid strategy combination")


class ModifiedRockPaperScissors(GameRules):
    def __init__(self) -> None:
        super().__init__()
        self.T = 1.3
        self.R = 1.0
        self.P = 0.1
        self.S = 0.0

        self.possible_strategies: List[int] = [1, 2, 3]

    def is_strategy_valid(self, strategy: int) -> bool:
        return strategy in self.possible_strategies

    def get_move(self, strategy: int) -> str:
        assert self.is_strategy_valid(strategy)

        if strategy == 1:
            return "R"
        if strategy == 2:
            return "P"
        if strategy == 3:
            return "S"

    def calculate_outcome(self, strategy_1: int, strategy_2: int) -> int:
        # Transform strategy to move in ['R', 'P', 'S'] for readability
        move_1 = self.get_move(strategy_1)
        move_2 = self.get_move(strategy_2)

        if move_1 == "R":
            if move_2 == "R":
                return self.R  # reward for cooperation
            if move_2 == "P":
                return self.S  # sucker's payoff
            if move_2 == "S":
                return self.T  # temptation to defect
        if move_1 == "P":
            if move_2 == "R":
                return self.T
            if move_2 == "P":
                return self.P  # punishment for mutual defection
            if move_2 == "S":
                return self.S
        if move_1 == "S":
            if move_2 == "R":
                return self.S
            if move_2 == "P":
                return self.T
            if move_2 == "S":
                return self.P

        raise ValueError("Invalid strategy combination")


class CollaborativeModifiedRPS(ModifiedRockPaperScissors):
    def calculate_outcome(self, strategy_1: int, strategy_2: int) -> int:
        # Transform strategy to move in ['R', 'P', 'S'] for readability
        move_1 = self.get_move(strategy_1)
        move_2 = self.get_move(strategy_2)

        if move_1 == "R":
            if move_2 == "R":
                return self.R  # reward for cooperation
            if move_2 == "P":
                return self.S  # sucker's payoff
            if move_2 == "S":
                return self.T  # temptation to defect
        if move_1 == "P":
            if move_2 == "R":
                return self.T
            if move_2 == "P":
                return self.R
            if move_2 == "S":
                return self.S
        if move_1 == "S":
            if move_2 == "R":
                return self.S
            if move_2 == "P":
                return self.T
            if move_2 == "S":
                return self.P

        raise ValueError("Invalid strategy combination")


class StableRPS(ModifiedRockPaperScissors):
    def calculate_outcome(self, strategy_1: int, strategy_2: int) -> int:
        # Transform strategy to move in ['R', 'P', 'S'] for readability
        move_1 = self.get_move(strategy_1)
        move_2 = self.get_move(strategy_2)

        if move_1 == "R":
            if move_2 == "R":
                return self.R  # reward for cooperation
            if move_2 == "P":
                return self.S  # sucker's payoff
            if move_2 == "S":
                return self.T  # temptation to defect
        if move_1 == "P":
            if move_2 == "R":
                return self.T
            if move_2 == "P":
                return self.P
            if move_2 == "S":
                return self.S
        if move_1 == "S":
            if move_2 == "R":
                return self.S
            if move_2 == "P":
                return self.P
            if move_2 == "S":
                return self.P

        raise ValueError("Invalid strategy combination")
