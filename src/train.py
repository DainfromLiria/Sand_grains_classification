from detector import MicroTextureDetector
from setup import setup

if __name__ == '__main__':
    setup()

    det = MicroTextureDetector(mode="train")
    det.train()

    # example of evaluation
    # det = MicroTextureDetector(mode="eval", experiment_uuid="03adda57-e689-4a1b-b1aa-84692602e12f")
    # det.evaluate_test_data(show_predictions=True)
