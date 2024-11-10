from dataset.augmentation import Augmentation
from dataset.dataset import SandGrainsDataset

if __name__ == '__main__':
    aug = Augmentation()
    aug.augment()
    # dt = SandGrainsDataset()
    # print(len(dt))

