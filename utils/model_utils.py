import torch
from models.model import LandmarkMapper

def load_mapping_network(config, device='cuda'):
    model = LandmarkMapper(config).to(device)

    if "PRETRAINED" in config["MODEL"]:
        checkpoint = torch.load(config["MODEL"]["PRETRAINED"])
        model.load_state_dict(checkpoint["state_dict"])

    return model