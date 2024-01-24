import numpy as np


class TestResult:
    n_elements: int
    times: list[float]

    def __init__(self,
                 n_elements: int = 0,
                 times: list[float] = []) -> None:
        self.n_elements = n_elements
        self.times = times

    def add_time(self, time: float) -> None:
        self.times.append(time)

    def get_statistics(self) -> tuple[float, float]:
        """returns mean and standard deviation of results"""
        return np.mean(self.times), np.std(self.times)
