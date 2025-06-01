from enum import Enum

class ToTSearchStrategy(Enum):
    BFS = "bfs"  # Breadth-First Search
    DFS = "dfs"  # Depth-First Search
    BEAM = "beam" # Beam Search

    def __str__(self):
        return self.value

class ToTScoringMethod(Enum):
    LLM = "llm"
    HEURISTIC = "heuristic"

    def __str__(self):
        return self.value
