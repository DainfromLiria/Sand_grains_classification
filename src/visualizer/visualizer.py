import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from configs import config


class Visualizer:

    def __init__(self):
        self.train_loss: List[float] = []
        self.validation_loss: List[float] = []

        self.iou: List[float] = []
        self.recall: List[float] = []
        self.precision: List[float] = []

        self.iou_per_class: List[List[float]] = []
        self.recall_per_class: List[List[float]] = []
        self.precision_per_class: List[List[float]] = []

    def visualize(self, results_folder_path: str):
        viz_path = os.path.join(results_folder_path, "visualizations")
        if not os.path.exists(viz_path):
            os.mkdir(viz_path)
        self.save_data(results_folder_path)
        self._create_loss_line_plots(viz_path)
        self._create_metrics_line_visualizations(viz_path)

    def save_data(self, results_folder_path: str):
        """Save train results to json file"""
        data = {
            'train_loss': self.train_loss,
            'validation_loss': self.validation_loss,
            'iou': self.iou,
            'recall': self.recall,
            'precision': self.precision,
            'iou_per_class': self.iou_per_class,
            'recall_per_class': self.recall_per_class,
            'precision_per_class': self.precision_per_class
        }
        data_path = os.path.join(results_folder_path, "data.json")
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=4)

    def load_data(self, results_folder_path: str):
        """Load train results from json file"""
        data_path = os.path.join(results_folder_path, "data.json")
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.train_loss = data['train_loss']
        self.validation_loss = data['validation_loss']
        self.iou = data['iou']
        self.recall = data['recall']
        self.precision = data['precision']
        self.iou_per_class = np.array(data['iou_per_class'])
        self.recall_per_class = np.array(data['recall_per_class'])
        self.precision_per_class = np.array(data['precision_per_class'])

    def _create_loss_line_plots(self, viz_path: str):
        assert len(self.train_loss) == len(self.validation_loss)

        epochs = list(range(len(self.train_loss)))

        plt.figure(figsize=(8, 6))
        sns.lineplot(x=epochs, y=self.train_loss, label="Train Loss")
        sns.lineplot(x=epochs, y=self.validation_loss, label="Validation Loss")

        plt.xticks(epochs)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train and Validation Loss per Epoch")
        plt.legend()
        plt.savefig(os.path.join(viz_path, "loss_line.png"))

    def store_metrics(self, avg_metrics: np.ndarray, per_class_metrics: np.ndarray):
        self.iou.append(avg_metrics[0].item())
        self.recall.append(avg_metrics[1].item())
        self.precision.append(avg_metrics[2].item())

        self.iou_per_class.append(per_class_metrics[0].tolist())
        self.recall_per_class.append(per_class_metrics[1].tolist())
        self.precision_per_class.append(per_class_metrics[2].tolist())

    def _create_metrics_line_visualizations(self, viz_path):
        assert len(self.iou) == len(self.recall) == len(self.precision)

        epochs = list(range(len(self.iou)))

        plt.figure(figsize=(8, 6))
        sns.lineplot(x=epochs, y=self.iou, label="IOU")
        sns.lineplot(x=epochs, y=self.recall, label="Recall")
        sns.lineplot(x=epochs, y=self.precision, label="Precision")

        plt.xticks(epochs)
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title("IOU, Recall and Precision per Epoch")
        plt.legend()
        plt.savefig(os.path.join(viz_path, "avg_metrics.png"))

    def create_per_class_metrics_visualizations(self, results_folder_path: str) -> None:
        self.load_data(results_folder_path=results_folder_path)
        with open(config.data.AUG_DATASET_INFO_PATH, "r") as file:
            classes_description_origin = json.load(file)["classes"]
            classes_description = {v: k for k, v in classes_description_origin.items()}
        classes_count = len(classes_description)
        epochs = list(range(len(self.iou_per_class)))
        viz_path = os.path.join(results_folder_path, "visualizations")

        for i in tqdm(range(classes_count), desc="Create visualizations"):
            plt.figure(figsize=(10, 8))
            sns.lineplot(x=epochs, y=self.iou_per_class[:, i], label="IOU")
            sns.lineplot(x=epochs, y=self.recall_per_class[:, i], label="Recall")
            sns.lineplot(x=epochs, y=self.precision_per_class[:, i], label="Precision")
            # plt.xticks(epochs)
            plt.xlabel("Epoch")
            plt.ylabel("Metric Value")
            plt.title(f"{classes_description[i]}")
            plt.legend()
            plt.savefig(os.path.join(viz_path, f"{classes_description[i]}.png"))
