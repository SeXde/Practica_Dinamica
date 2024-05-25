from datetime import datetime
import numpy as np
from src import step


class Pipeline:
    """
    A class to represent a processing pipeline composed of multiple steps.

    Attributes:
    -----------
    name : str
        Name of the pipeline.
    steps : list of step
        List of steps in the pipeline.
    times : list of float
        List to store execution time of each step.
    mute : bool
        If True, suppresses print statements. Defaults to True.
    default_output : tuple
        Default output if a step returns None. Defaults to (np.inf, np.inf).

    Methods:
    --------
    run(input_params):
        Runs the pipeline on the given input parameters.
    """

    def __init__(self, name: str, steps: list[step], mute: bool = True, default_output=(np.inf, np.inf)):
        """
        Initializes the Pipeline with the given name, steps, and settings.

        Parameters:
        -----------
        name : str
            Name of the pipeline.
        steps : list of step
            List of steps in the pipeline.
        mute : bool, optional
            If True, suppresses print statements. Defaults to True.
        default_output : tuple, optional
            Default output if a step returns None. Defaults to (np.inf, np.inf).
        """
        self.name = name
        self.steps = steps
        self.times = []
        self.mute = mute
        self.default_output = default_output

        if not self.mute:
            print(f'`{self.name}` has been successfully created with {len(self.steps)} steps.')

    def run(self, input_params):
        """
        Runs the pipeline on the given input parameters.

        Parameters:
        -----------
        input_params : any
            The input parameters to be processed by the pipeline.

        Returns:
        --------
        tuple
            The output of the pipeline and a boolean indicating success (True) or failure (False).
        """
        start_time = datetime.now()

        for current_step in self.steps:
            step_time = datetime.now()
            input_params = current_step.run(input_params)
            step_elapsed = (datetime.now() - step_time).total_seconds()
            self.times.append(step_elapsed)

            if input_params is None:
                return self.default_output, False

        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        if not self.mute:
            print(f'Pipeline `{self.name}` completed in {elapsed_time:.2f} seconds')

        self.times = np.array(self.times)
        arg_max = np.argmax(self.times)

        if not self.mute:
            print(f'The slowest step was `{self.steps[arg_max].step_name}` with {self.times[arg_max]} seconds')

        self.times = []  # Reset times after the run
        return input_params, True
