from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    def plot_evaluation(self, save=True):
        pass
