import os
from dotenv import load_dotenv


# ENV variables
load_dotenv()
NEPTUNE_PROJECT=os.getenv("NEPTUNE_PROJECT")
NEPTUNE_API_KEY=os.getenv("NEPTUNE_API_KEY")


if __name__ == '__main__':
    print(NEPTUNE_PROJECT)