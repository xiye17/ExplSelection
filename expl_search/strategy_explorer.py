import random
import numpy as np

from abc import ABC
from tqdm import tqdm




class StrategyExplorerBase(ABC):
    def __init__(self, strategy_name, top_candidates):
        self.strategy_name = strategy_name
        self.top_candidates = top_candidates
        self.step_cursor = 0

    @classmethod
    def from_strategy_name(cls, strategy_name, top_candidates):
        if strategy_name == "dummy":
            return DummyStrategyExplorer(strategy_name, top_candidates)
        elif strategy_name.startswith("randtop"):
            return RandTopStrategyExplorer(strategy_name, top_candidates)
        else:
            raise RuntimeError("Not Implemented Yet")

    def tick(self):
        this_cursor = self.step_cursor
        self.step_cursor = self.step_cursor + 1
        return this_cursor

    def init_status(self, top_candidates):
        raise NotImplementedError("Init status not implemetned " + self.__class__.__name__)

    def get_next_combo(explored_combos):
        raise NotImplementedError("Get next combo not implemetned " + self.__class__.__name__)

class DummyStrategyExplorer(StrategyExplorerBase):
    def init_status(self, args):
        self.step_cursor = 0

    def get_next_combo(self, explored_combos):        
        cursor = self.tick()
        return cursor, self.top_candidates[cursor]

class RandTopStrategyExplorer(StrategyExplorerBase):
    def __init__(self, strategy_name, top_candidates):
        super().__init__(strategy_name, top_candidates)
        self.rand_perms = None
        num_top = strategy_name[len("randtop"):]
        if not num_top:
            num_top = len(top_candidates)
        else:
            num_top = int(num_top)
        assert num_top <= len(top_candidates)
        self.num_top = num_top

    def init_status(self, args):
        self.step_cursor = 0
        np.random.seed(args.randseed)
        self.rand_perms = np.random.permutation(self.num_top)

    def get_next_combo(self, explored_combos):
        cursor = self.tick()
        rand_idx = self.rand_perms[cursor]
        return rand_idx, self.top_candidates[rand_idx]
