from abc import ABC, abstractmethod


class Step(ABC):
    def __init__(self, step_name: str):
        self.step_name = step_name

    @abstractmethod
    def run(self, inputs):
        pass
