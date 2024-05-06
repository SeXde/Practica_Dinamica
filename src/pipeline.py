from datetime import datetime
import numpy as np
from src import step


class Pipeline:

    def __init__(self, name: str, steps: list[step], mute: bool = True):
        self.steps = steps
        self.times = []
        self.name = name
        self.mute = mute
        if not self.mute:
            print(f'`{self.name}` has been successfully created with {len(self.steps)} steps.')

    def run(self, input_params):
        start_time = datetime.now()
        for current_step in self.steps:
            step_time = datetime.now()
            input_params = current_step.run(input_params)
            step_elapsed = (datetime.now() - step_time).total_seconds()
            self.times.append(step_elapsed)
            if input_params is None:
                break
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        if not self.mute:
            print(
                f'Pipeline ´{self.name}´ is completed in {elapsed_time:.2f} seconds')
        self.times = np.array(self.times)
        arg_max = np.argmax(self.times)
        if not self.mute:
            print(
                f'The slowest step was ´{self.steps[arg_max].step_name}´ with {self.times[arg_max]} seconds'
            )
        self.times = []
        return input_params
