import logging

import torch

from configs.config import config
# from dataset.augmentation import Augmentation
# from dataset.dataset import SandGrainsDataset
from detector.detector import MicroTextureDetector
# from metrics.loss import FocalLoss, FocalTverskyLoss
# from metrics.metrics import calculate_metrics, convert_nn_output
from utils.logging import setup_logging
# from visualizer.visualizer import Visualizer

setup_logging()

logger = logging.getLogger(__name__)

predict = [
    [
        [
            [1, 1], [0, 0]
        ],
        [
            [0, 1], [1, 1]
        ],
        [
            [1, 1], [1, 1]
        ]
    ],
    [
        [
            [1, 1], [0, 0]
        ],
        [
            [0, 1], [1, 1]
        ],
        [
            [1, 1], [1, 1]
        ]
    ]
]
real = [
    [
        [
            [1., 0.], [1., 1.]
        ],
        [
            [0., 0.], [0., 1.]
        ],
        [
            [1., 1.], [1., 0.]
        ]
    ],
    [
        [
            [1., 0.], [1., 1.]
        ],
        [
            [0., 0.], [0., 1.]
        ],
        [
            [1., 1.], [1., 0.]
        ]
    ]
]


def metric_tests():
    pred = torch.tensor(data=[
        [
            [
                [1.5, 2.5], [-0.781, -0.743]
            ],
            [
                [-2.56, 0], [0.89, 1]
            ],
            [
                [0.87463, 0.23], [1.7, 1.89]
            ]
        ],
        [
            [
                [1.5, 2.5], [-0.781, -0.743]
            ],
            [
                [-2.56, 0], [0.89, 1]
            ],
            [
                [0.87463, 0.23], [1.7, 1.89]
            ]
        ]
    ], dtype=torch.float32)

    target = torch.tensor(data=[
        [
            [
                [1, 0], [1, 1]
            ],
            [
                [0, 0], [0, 1]
            ],
            [
                [1, 1], [1, 0]
            ]
        ],
        [
            [
                [1, 0], [1, 1]
            ],
            [
                [0, 0], [0, 1]
            ],
            [
                [1, 1], [1, 0]
            ]
        ]
    ], dtype=torch.float32)
    # f_loss = FocalLoss()
    # f_loss = CrossEntropyLoss()
    # loss = f_loss(pred, target)
    # logger.info(f'Loss: {loss}')
    # calculate_metrics(pred, target)
    # convert_nn_output(outputs=pred, to_mask=True)
    # ft_loss = FocalTverskyLoss()
    # loss = ft_loss(pred, target)
    # print(loss)


if __name__ == '__main__':
    torch.hub.set_dir(config.model.MODELS_DIR_PATH)  # TODO move to some setup file
    # aug = Augmentation()
    # aug.augment()

    # dt = SandGrainsDataset(path=config.data.AUG_TRAIN_SET_PATH)
    # img, masks = dt[0]
    # logger.info(f"Image shape: {img.shape} - Mask shape: {masks.shape}")

    net = MicroTextureDetector()
    net.train()

    # metric_tests()

    # viz = Visualizer()
    # viz.create_per_class_metrics_visualizations("../results/FocalLoss_300_epochs_12_batches_1e-05_lr")

    # res_folder = "FocalLoss_300_epochs_5_batches_0.001_lr_without_normalization_without_resize"
    # res_folder = "FocalLoss_300_epochs_16_batches_1e-05_lr_with_normalization"
    # net_test = MicroTextureDetector(model_path=f"../results/{res_folder}/model.pt")
    # net_test.evaluate_test_data(results_folder_path=f"../results/{res_folder}")
    # net_test.predict("A3_44.tif")
