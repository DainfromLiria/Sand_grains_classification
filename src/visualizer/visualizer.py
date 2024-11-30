import json
import os
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
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

        # read classes descriptions
        with open(config.data.AUG_DATASET_INFO_PATH, "r") as file:
            classes_description_origin = json.load(file)["classes"]
            self.classes_description: Dict[int, str] = {v: k for k, v in classes_description_origin.items()}

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
        epochs = list(range(len(self.iou_per_class)))
        viz_path = os.path.join(results_folder_path, "visualizations")
        classes_count = len(self.classes_description)

        for i in tqdm(range(classes_count), desc="Create visualizations"):
            plt.figure(figsize=(10, 8))
            sns.lineplot(x=epochs, y=self.iou_per_class[:, i], label="IOU")
            sns.lineplot(x=epochs, y=self.recall_per_class[:, i], label="Recall")
            sns.lineplot(x=epochs, y=self.precision_per_class[:, i], label="Precision")
            # plt.xticks(epochs)
            plt.xlabel("Epoch")
            plt.ylabel("Metric Value")
            plt.title(f"{self.classes_description[i]}")
            plt.legend()
            plt.savefig(os.path.join(viz_path, f"{self.classes_description[i]}.png"))

    def make_test_image_prediction_visualisations(
            self,
            image: torch.Tensor,
            masks: torch.Tensor,
            outputs: torch.Tensor,
            results_folder_path: str

    ) -> None:
        image_rgb = cv2.cvtColor(torch.squeeze(image).cpu().numpy(), cv2.COLOR_GRAY2RGB)
        classes_count = len(self.classes_description)

        viz_folder = os.path.join(results_folder_path, "visualizations")
        if not os.path.exists(viz_folder):
            os.mkdir(viz_folder)
        img_folder = os.path.join(viz_folder, "images")
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

        for i in range(classes_count):
            mask_rgb = np.stack((masks[i].cpu().numpy(),) * 3, axis=-1)
            output_rgb = np.stack((outputs[i].cpu().numpy(),) * 3, axis=-1)
            if len(np.unique(mask_rgb)) > 1 or len(np.unique(output_rgb)) > 1:
                # show on one frame
                masked_img = np.copy(image_rgb)
                masked_img[(mask_rgb == 1.0).all(-1)] = [0, 255, 0]
                masked_img[(output_rgb == 1.0).all(-1)] = [0, 0, 255]
                result = cv2.addWeighted(masked_img, 0.3, image_rgb, 0.7, 0, masked_img)
                img_path = os.path.join(img_folder, f"{self.classes_description[i]}.png")
                cv2.imwrite(img_path, result)
                # show on two separate frames
                masked_img_duo1 = np.copy(image_rgb)
                masked_img_duo2 = np.copy(image_rgb)
                masked_img_duo1[(mask_rgb == 1.0).all(-1)] = [0, 255, 0]
                result1 = cv2.addWeighted(masked_img_duo1, 0.3, image_rgb, 0.7, 0, masked_img_duo1)
                masked_img_duo2[(output_rgb == 1.0).all(-1)] = [0, 0, 255]
                result2 = cv2.addWeighted(masked_img_duo2, 0.3, image_rgb, 0.7, 0, masked_img_duo2)
                concatenated = np.hstack((result1, result2))
                img_path = os.path.join(img_folder, f"{self.classes_description[i]}_duo.png")
                cv2.imwrite(img_path, concatenated)

    def make_prediction_visualisation(self, image: torch.Tensor, pred: np.ndarray, out_folder_path: str) -> None:
        classes_count = len(self.classes_description)
        image_rgb = cv2.cvtColor(torch.squeeze(image).cpu().numpy(), cv2.COLOR_GRAY2RGB)
        for i in range(classes_count):
            pred_rgb = np.stack((pred[i],) * 3, axis=-1)
            if len(np.unique(pred_rgb)) > 1:
                masked_img = np.copy(image_rgb)
                masked_img[(pred_rgb == 1.0).all(-1)] = [0, 255, 0]
                result = cv2.addWeighted(masked_img, 0.3, image_rgb, 0.7, 0, masked_img)
                img_path = os.path.join(out_folder_path, f"{self.classes_description[i]}.png")
                cv2.imwrite(img_path, result)
