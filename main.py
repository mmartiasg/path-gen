from path_gen.data.loader import TrackXYLoader
from path_gen.models.regressive import RegressiveLSTM
from path_gen.training.trainer import BaseTrainer
import torch
from path_gen.training.losses import MSELoss

def main():
    dataset = torch.utils.data.DataLoader(TrackXYLoader("/Users/matiasgonzalez/workspace/crab_track/0.4.6-tracks/preds/yolo_kalman/12.txt"))
    model = RegressiveLSTM()
    trainer = BaseTrainer(epochs=100, loss=MSELoss())
    loss = trainer.train(model, dataset)
    print(loss)

if __name__ == "__main__":
    main()
