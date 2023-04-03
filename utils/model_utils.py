import torch
from models.model import LandmarkMapper

def load_mapping_network(config, device='cuda'):
    model = LandmarkMapper(config).to(device)

    if "PRETRAINED" in config["MODEL"]:
        checkpoint = torch.load(config["MODEL"]["PRETRAINED"])
        # Remove the "module." prefix from the parameter names of the saved state_dict
        new_state_dict = {}
        for key, value in checkpoint["state_dict"].items():
            if key.startswith('module.'):
                new_key = key[7:]  # remove the "module." prefix
            else:
                new_key = key
            new_state_dict[new_key] = value

        # Load the modified state_dict into the model's state_dict
        model.load_state_dict(new_state_dict)

    return model