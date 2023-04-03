import argparse
from yaml import safe_load
import os
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision import utils
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from logger import create_logger
from tensorboardX import SummaryWriter
from evaluate import AverageMeter


from models.model import LandmarkMapper
from data_parsing.landmarks_dataset import LandmarksDataset
from logger import create_logger
from utils.plotting_utils import plot_points_on_image


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    parser.add_argument(
        "--config", help="training configuration file", required=True, type=str
    )

    return parser.parse_args()

class Trainer:
    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.config = self.parse_config(config_file)

        self.batch_size = self.config["TRAIN_SETUP"]["BATCH_SIZE"]

        current_time = datetime.now()
        self.save_location = os.path.join(
            'weights', 'experiments', current_time.strftime("%d_%m_%Y__%H_%M_%S")
        )
        os.makedirs(self.save_location, exist_ok=True)

        self.model = LandmarkMapper(self.config)
        gpus = self.config["TRAIN_SETUP"]["DEVICES"]
        self.model = torch.nn.DataParallel(self.model, device_ids=gpus).cuda()

        self.train_dataset = LandmarksDataset(self.config["DATASET_TRAIN"]["PATH"])
        sampler = RandomSampler(
            self.train_dataset,
            replacement=True,
            num_samples=self.config["TRAIN_SETUP"]["STEPS_PER_EPOCH"] * self.config["TRAIN_SETUP"]["BATCH_SIZE"]
        )
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.config["TRAIN_SETUP"]["WORKERS"]
        )

        self.valid_dataset = LandmarksDataset(self.config["DATASET_VALID"]["PATH"], return_image=True)
        self.valid_loader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=self.config["DATASET_VALID"]["SHUFFLE"]
        )

        self.loss = torch.nn.MSELoss()

        print("Number of trainable parameters:")
        print(len(list(filter(lambda p: p.requires_grad, self.model.parameters()))))

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config["TRAIN_SETUP"]["LEARNING_RATE"]
        )

        self.scaler = GradScaler()


    def parse_config(self, config_file: str) -> dict:
        config = {}
        if os.path.isfile(config_file) and config_file.endswith(".yml"):
            with open(config_file, "r") as f_config:
                config = safe_load(f_config) 
        else:
            print("Invalid config path: {}".format(config_file))
        
        return config
    
    def save_checkpoint(self, states: dict, filename: str) -> None:
        torch.save(states, os.path.join(self.save_location, filename))

    def train_step(
        self,
        landmarks3d: torch.Tensor,
        landmarks2d: torch.Tensor
    ) -> list([torch.Tensor, torch.Tensor]):
        
        self.optimizer.zero_grad(set_to_none=True)
        
        preds = self.model(landmarks3d)
        loss = self.loss(preds, landmarks2d)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss, preds

    def train_epoch(self, epoch: int, writer_dict: dict) -> str:
        losses = AverageMeter()

        self.model.train()
        for landmarks3d, landmarks2d in tqdm(self.train_loader):
            landmarks3d = landmarks3d.cuda(non_blocking=True)
            landmarks2d = landmarks2d.cuda(non_blocking=True)

            loss, preds = self.train_step(landmarks3d, landmarks2d)

            losses.update(loss, preds.size(0))

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
        
        train_result = 'Train Epoch {} loss:{:.4f}'.format(epoch, losses.avg)
        return train_result

    def valid_step(
        self,
        landmarks3d: torch.Tensor,
        landmarks2d: torch.Tensor
    ) -> list([torch.Tensor, torch.Tensor]):
        preds = self.model(landmarks3d)
        loss = self.loss(preds, landmarks2d)

        return loss, preds

    def validate(self, epoch: int, writer_dict: dict) -> str:
        losses = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for landmarks3d, landmarks2d, image in tqdm(self.valid_loader):
                landmarks3d = landmarks3d.cuda(non_blocking=True)
                landmarks2d = landmarks2d.cuda(non_blocking=True)
                loss, preds = self.valid_step(landmarks3d, landmarks2d)

                losses.update(loss.item(), preds.size(0))
        
        valid_result = (
            "Valid Epoch {} loss:{:.4f}".format(epoch, losses.avg)
        )

        image = image[0].numpy().astype(np.uint8)
        pred_points2d = preds[0].detach().cpu().numpy()
        input_points3d = landmarks3d[0].detach().cpu().numpy()
        image_with_points = plot_points_on_image(image, pred_points2d, input_points3d)

        image_with_points = torch.from_numpy(image_with_points)
        image_with_points = image_with_points.permute((2, 0, 1))

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_image('image_with_points', image_with_points, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
        writer.flush()

        return valid_result

    def fit(self) -> None:
        torch.backends.cudnn.benchmark = True

        logger, tensorboard_log_dir = create_logger(self.save_location)
        writer_dict = {
            'writer': SummaryWriter(log_dir=tensorboard_log_dir, flush_secs=5),
            'train_global_steps': 0,
            'valid_global_steps': 0
        }

        if self.config["TRAIN_SETUP"]["RESUME"]:
            model_state_file = self.config["TRAIN_SETUP"]["RESUME_CHECKPOINT"]
            if os.path.is_file(model_state_file):
                checkpoint = torch.load(model_state_file)
                self.config["TRAIN_SETUP"]["BEGIN_EPOCH"] = checkpoint["epoch"]
                if "state_dict" in checkpoint.keys():
                    state_dict = checkpoint["state_dict"]
                    self.model.load_state_dict(state_dict)
                else:
                    self.model.module.load_state_dict(model_state_file)
                
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                print("Error: no checkpoint found")
        
        for epoch in tqdm(range(self.config["TRAIN_SETUP"]["BEGIN_EPOCH"], self.config["TRAIN_SETUP"]["END_EPOCH"])):
            logger.info("Training epoch: {}".format(epoch))
            train_result = self.train_epoch(epoch, writer_dict)
            logger.info(train_result)

            if ((epoch + 1) % self.config["TRAIN_SETUP"]["SAVE_FREQ"]) == 0:
                valid_result = self.validate(epoch, writer_dict)
                logger.info(valid_result)
                
                self.save_checkpoint(
                    {
                        "state_dict": self.model.state_dict(),
                        "epoch": epoch + 1,
                        "optimizer": self.optimizer.state_dict()
                    },
                    'checkpoint_{}.pth'.format(epoch)
                )
            
            writer_dict['writer'].flush()
            
        writer_dict['writer'].close()

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(config_file=args.config)
    trainer.fit()
