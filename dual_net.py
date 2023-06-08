import torch
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])

if torch.cuda.is_available():
    args.device = 'cuda:0'


class DualNet(torch.nn.Module):
    def __init__(self, args):
        super(DualNet, self).__init__()

        self.conv1 = ChebConv(1025, 512, 2)
        self.conv2 = ChebConv(512, 512, 2)

        self.lin01 = torch.nn.Linear(1024, 512)
        self.lin02 = torch.nn.Linear(512, 256)
        self.lin03 = torch.nn.Linear(256, 2)

        self.lin11 = torch.nn.Linear(1024, 512)
        self.lin12 = torch.nn.Linear(512, 256)
        self.lin13 = torch.nn.Linear(256, 4)

    def forward(self, data, flag):
        x, edge_index, batch, A = data.x, data.edge_index, data.batch, data.A
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        feature0 = x
        # Binary output
        x0 = F.relu(self.lin01(x))
        x0 = F.dropout(x0, p=0.5, training=self.training)

        x0 = F.relu(self.lin02(x0))
        x0 = F.dropout(x0, p=0.5, training=self.training)

        x0 = F.relu(self.lin03(x0))
        output0 = F.softmax(x0, dim=-1)
        # Class-wise output
        x1 = F.relu(self.lin11(x))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x1 = F.relu(self.lin12(x1))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        feature1 = x1

        x1 = F.relu(self.lin13(x1))
        output1 = F.softmax(x1, dim=-1)

        return output0, output1, feature0, feature1


def main():
    model = DualNet(args).to(args.device)


if __name__ == '__main__':
     main()




