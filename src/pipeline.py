from datetime import datetime

from src import step


class Pipeline:

    def __init__(self, name: str, steps: list[step]):
        self.steps = steps
        self.name = name
        print(f'`{self.name}` has been successfully created with {len(self.steps)} steps.')

    def run(self, input_params):
        start_time = datetime.now()
        for current_step in self.steps:
            input_params = current_step.run(*input_params)
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        print(
            f'Pipeline ´{self.name}´ is completed in {elapsed_time:.2f} seconds')
        return input_params
