from torch import nn

class LandmarkMapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config["MODEL"]["NUM_POINTS"] * 2
        self.fc1 = nn.Linear(self.dim, self.dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.dim, self.dim)

    def forward(self, landmarks3d):
        x = self.fc1(landmarks3d)
        x = self.relu(x)
        landmarks2d = self.fc2(x)

        return landmarks2d
