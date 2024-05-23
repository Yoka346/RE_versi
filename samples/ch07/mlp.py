import torch
from torch import Tensor
from torch import nn

from pyrev import Position, BoardCoordinate


INPUT_SIZE = 2 * 8 * 8
HIDDEN_SIZE = 16


def positions_to_batch(positions: list[Position], dest: Tensor=None):
    """
    局面のリストを作って, MLPに入力するバッチを作成する
    """
    batch_size = len(positions)

    if dest is None:
        dest = torch.empty(size=(INPUT_SIZE, batch_size), dtype=torch.float32)
    
    dest.fill_(0.0)

    if dest.shape[1] != batch_size:
        raise ValueError("The shape of dest must agree (INPUT_SIZE, batch_size).")
    
    for batch_id in range(batch_size):
        pos = positions[batch_id]
        for coord in pos.get_player_disc_coords():
            dest[coord][batch_size] = 1.0

        offset = 64
        for coord in pos.get_opponent_disc_coords():
            dest[offset + coord][batch_id] = 1.0

    return dest


class ValueNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ValueNetwork, self).__init__()
        self.__seq = nn.Sequential(
           nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
           nn.ReLU(),
           nn.Linear(512, 1),
           nn.Tanh()
           )

    def forward(self, x):
        return self.__seq(x)


class QNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(QNetwork, self).__init__()
        self.__seq = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 64),
            nn.Tanh()
        )

    def forward(self, x):
        return self.__seq(x)
    

class PolicyNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PolicyNetwork, self).__init__()
        self.__seq = nn.Sequential(
            nn.Linear(2 * 8 * 8, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 64),
            nn.LogSoftmax()
        )

    def forward(self, x):
        return self.__seq(x)
