import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class PipelineEvaluator:
    """
    Class to evaluate multiple pipelines.

    Attributes:
    -----------
    pipeline : dict
        A dictionary mapping pipeline names to their respective data.
    y : numpy.ndarray
        The reference points.

    Methods:
    --------
    __init__(x, y, pipeline_names):
        Initializes the class with the pipeline data, reference points, and pipeline names.
    show_evaluation(save=True):
        Displays a plot showing the distance between points in x and y for each pipeline.
    """

    def __init__(self, x, y, pipeline_names):
        """
        Initializes the PipelineEvaluator class.

        Parameters:
        -----------
        x : numpy.ndarray
            An array of shape (n, m, k) where:
                - n is the number of pipelines,
                - m is the number of points,
                - k is the dimension of the points.
        y : numpy.ndarray
            An array of shape (m, k) representing the reference points.
        pipeline_names : list of str
            A list of names for the pipelines.

        Raises:
        -------
        AssertionError:
            If the dimensions of x or y are not as expected.
        """

        assert len(x.shape) == 3, (
            'X shape should be (n, m, k) where n is the number of pipelines, '
            'm is the number of points, and k is the dimension of the points'
        )
        assert x.shape[1:] == y.shape, (
            'Y shape should be (m, k) where m is the number of points and k is the dimension of the points'
        )
        assert x.shape[0] == len(pipeline_names), (
            'The number of pipelines in X should match the number of provided pipeline names'
        )

        self.pipeline = {name: pipeline_x for pipeline_x, name in zip(x, pipeline_names)}
        self.y = y

    def show_evaluation(self, save=True, infinite_distance=10):
        """
        Displays a plot showing the distance between points in x and y for each pipeline,
        and a bar plot showing the pipeline with the lowest accumulated distance.

        Parameters:
        -----------
        save : bool
            If True, the plot will be saved as 'evaluation_plot.png'. Defaults to True.
        """
        plt.figure(figsize=(14, 10))

        # Prepare the data for plotting
        data = []
        penalized_distances = []
        for name, pipeline_x in self.pipeline.items():
            for i, (point_x, point_y) in enumerate(zip(pipeline_x, self.y)):
                if np.any(np.isinf(point_x)):
                    distance = infinite_distance  # Penalize pipelines with undetected points
                    penalized_distances.append([name, i, distance])
                else:
                    distance = np.linalg.norm(point_x - point_y)
                    data.append([name, i, distance])

        # Convert to DataFrame for seaborn
        df = pd.DataFrame(data, columns=['Pipeline', 'Frame', 'Distance'])

        # Plot the line plot using seaborn
        plt.subplot(2, 1, 1)
        sns.lineplot(data=df, x='Frame', y='Distance', hue='Pipeline', marker='o')

        # Add horizontal line at y = 0
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)

        # Add title and labels
        plt.title('Distance per Frame for Each Pipeline')
        plt.xlabel('Frame')
        plt.ylabel('Distance')
        plt.legend()

        # Calculate accumulated distances including penalties
        all_distances = data + penalized_distances
        df_all = pd.DataFrame(all_distances, columns=['Pipeline', 'Frame', 'Distance'])
        accumulated_distances = df_all.groupby('Pipeline')['Distance'].sum().reset_index()

        # Sort by the accumulated distance
        accumulated_distances = accumulated_distances.sort_values(by='Distance')

        # Update pipeline names with ranking
        accumulated_distances['Pipeline'] = [f'#{idx + 1} {name}' for idx, name in enumerate(accumulated_distances['Pipeline'])]

        # Plot the bar plot
        plt.subplot(2, 1, 2)
        sns.barplot(data=accumulated_distances, x='Pipeline', y='Distance', palette='viridis')

        # Add title and labels
        plt.title('Accumulated Distance per Pipeline')
        plt.xlabel('Pipeline')
        plt.ylabel('Accumulated Distance')

        if save:
            plt.savefig('evaluation_plot.png')

        plt.tight_layout()
        plt.show()


# Example usage:
x = np.random.rand(3, 100, 2)  # 3 pipelines, 100 points, 2 dimensions
x[0, 10] = np.inf  # Example of a point not detected in pipeline 1 at frame 10
y = np.random.rand(100, 2)  # 100 points, 2 dimensions
pipeline_names = ['Kalman', 'PF', 'Other']
evaluator = PipelineEvaluator(x, y, pipeline_names)
evaluator.show_evaluation()
