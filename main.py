from src.utils import read_options
from src.trainers.base import Trainer

if __name__ == '__main__':
    params = read_options()
    trainer = Trainer(params)
    trainer.train()
