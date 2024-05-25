from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from evaluator import Evaluator


class PipelineEvaluator(Evaluator):
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

    def plot_evaluation(self, save=True, infinite_distance=10):
        """
        Displays a plot showing the distance between points in x and y for each pipeline,
        and a bar plot showing the pipeline with the lowest accumulated distance.

        Parameters:
        -----------
        save : bool
            If True, the plot will be saved as 'pipeline_evaluation_plot.png'. Defaults to True.
        infinite_distance : float
            Distance value to use for penalizing pipelines with undetected points. Defaults to 10.
        """
        plt.figure(figsize=(14, 10))

        # Prepare the data for plotting
        data = []
        penalized_distances = []
        inf_data = defaultdict(list)
        for name, pipeline_x in self.pipeline.items():
            for i, (point_x, point_y) in enumerate(zip(pipeline_x, self.y)):
                if np.any(np.isinf(point_x)):
                    distance = infinite_distance  # Penalize pipelines with undetected points
                    penalized_distances.append([name, i, distance])
                    inf_data[name].append((i, distance))
                else:
                    distance = np.linalg.norm(point_x - point_y)
                    data.append([name, i, distance])

        # Convert to DataFrame for seaborn
        df = pd.DataFrame(data, columns=['Pipeline', 'Frame', 'Distance'])

        # Define color palette
        palette = sns.color_palette("tab10", len(self.pipeline))
        color_mapping = {name: color for name, color in zip(self.pipeline.keys(), palette)}

        # Plot the line plot using seaborn
        plt.subplot(2, 1, 1)
        sns.lineplot(data=df, x='Frame', y='Distance', hue='Pipeline', marker='o', palette=color_mapping)

        # Plot the points with infinite distances as 'x' at y = 0
        for name, points in inf_data.items():
            for (frame, _) in points:
                plt.plot(frame, 0, 'x', color=color_mapping[name])

        # Add horizontal line at y = 0
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)

        # Add title and labels
        plt.title('Distance per Frame for Each Pipeline')
        plt.xlabel('Frame')
        plt.ylabel('Distance')

        # Add custom legend
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color='black', marker='x', linestyle='None'))
        labels.append('Undetected Points')
        plt.legend(handles=handles, labels=labels, loc='upper right')

        # Calculate accumulated distances including penalties
        all_distances = data + penalized_distances
        df_all = pd.DataFrame(all_distances, columns=['Pipeline', 'Frame', 'Distance'])
        accumulated_distances = df_all.groupby('Pipeline')['Distance'].sum().reset_index()

        # Preserve original order for coloring
        accumulated_distances['Color'] = accumulated_distances['Pipeline'].map(color_mapping)
        sorted_pipelines = accumulated_distances.sort_values(by='Distance')['Pipeline']

        # Plot the bar plot
        plt.subplot(2, 1, 2)
        sns.barplot(data=accumulated_distances, x='Pipeline', y='Distance', palette=color_mapping, order=sorted_pipelines)

        # Add title and labels
        plt.title('Accumulated Distance per Pipeline')
        plt.xlabel('Pipeline')
        plt.ylabel('Accumulated Distance')

        if save:
            plt.savefig('pipeline_evaluation_plot.png')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Example usage:
    x = np.random.rand(3, 100, 2)  # 3 pipelines, 100 points, 2 dimensions
    x[0, 10] = np.inf  # Example of a point not detected in pipeline 1 at frame 10
    x[1, 23] = np.inf
    x[2, 63] = np.inf
    y = np.random.rand(100, 2)  # 100 points, 2 dimensions
    pipeline_names = ['Kalman', 'PF', 'Other']
    evaluator = PipelineEvaluator(x, y, pipeline_names)
    evaluator.plot_evaluation(save=False)
