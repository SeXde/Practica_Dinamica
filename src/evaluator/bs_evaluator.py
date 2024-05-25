from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

from evaluator import Evaluator


class BSEvaluator(Evaluator):
    """
    Class to evaluate multiple background subtraction methods.

    Attributes:
    -----------
    bs : dict
        A dictionary mapping background subtraction method names to their respective data.
    y : numpy.ndarray
        The reference images.

    Methods:
    --------
    __init__(x, y, bs_names):
        Initializes the class with the background subtraction data, reference images, and method names.
    plot_evaluation():
        Plots precision, recall, and F1-score for each background subtraction method.
    __get_precision_and_recall(estimation, ground_truth):
        Calculates precision and recall for a single pair of estimation and ground truth images.
    """

    def __init__(self, x, y, bs_names):
        """
        Initializes the BSEvaluator class.

        Parameters:
        -----------
        x : numpy.ndarray
            An array of shape (n, m, h, w) where:
                - n is the number of background subtraction methods,
                - m is the number of images,
                - h is the height of the image, and
                - w is the width of the image.
        y : numpy.ndarray
            An array of shape (m, h, w) representing the reference images.
        bs_names : list of str
            A list of names for the background subtraction methods.

        Raises:
        -------
        AssertionError:
            If the dimensions of x or y are not as expected.
        """

        assert len(x.shape) == 4, (
            'X shape should be (n, m, h, w) where n is the number of bs methods, '
            'm is the number of images, h is the height of the image, and w is the width'
        )
        assert x.shape[1:] == y.shape, (
            'Y shape should be (m, h, w) where m is the number of images, '
            'h is the height of the image, and w is the width'
        )
        assert x.shape[0] == len(bs_names), (
            'The number of bs methods in X should match the number of provided bs method names'
        )

        self.bs = {name: bs_x for bs_x, name in zip(x, bs_names)}
        self.y = y

    def plot_evaluation(self, save=True):
        """
        Plots average precision, recall, and F1-score for each background subtraction method.
        """
        data = defaultdict(list)

        for name, pipeline_x in self.bs.items():
            precisions = []
            recalls = []
            f1_scores = []
            for image_x, image_y in zip(pipeline_x, self.y):
                precision, recall, f1 = self.__get_precision_recall_and_f1(image_x, image_y)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1 = np.mean(f1_scores)
            data['Method'].append(
                f'{name}\n(Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, F1: {avg_f1:.2f})')
            data['Precision'].append(avg_precision)
            data['Recall'].append(avg_recall)
            data['F1 Score'].append(avg_f1)

        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))

        sns.barplot(data=df.melt(id_vars='Method'), x='Method', y='value', hue='variable', palette='muted')
        plt.title('Background Subtraction Evaluation')
        plt.xlabel('Method')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(title='Metric')

        if save:
            plt.savefig('bs_evaluation_plot.png')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def __get_precision_recall_and_f1(estimation, ground_truth, epsilon=1e-20):
        """
        Calculates precision and recall for a single pair of estimation and ground truth images.

        Parameters:
        -----------
        estimation : numpy.ndarray
            Estimated binary image.
        ground_truth : numpy.ndarray
            Ground truth binary image.

        Returns:
        --------
        precision : float
            Precision value.
        recall : float
            Recall value.
        """
        assert estimation.dtype == np.uint8
        assert ground_truth.dtype == estimation.dtype

        precision, recall, f1, _ = precision_recall_fscore_support(ground_truth.flatten(), estimation.flatten(),
                                                                   labels=[0, 255], zero_division=True)

        return precision[0], recall[0], f1[0]


if __name__ == "__main__":
    x = np.random.randint(0, 2, size=(3, 10, 10, 10), dtype=np.uint8) * 255
    # x = np.zeros((3, 10, 10, 10), dtype=np.uint8)
    y = np.random.randint(0, 2, size=(10, 10, 10), dtype=np.uint8) * 255
    # y = np.zeros((10, 10, 10), dtype=np.uint8)
    bs_names = ['Mean bs', 'Mode bs', 'GMM bs']

    evaluator = BSEvaluator(x, y, bs_names)
    evaluator.plot_evaluation(save=False)
