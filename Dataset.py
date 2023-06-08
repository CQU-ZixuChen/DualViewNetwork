import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
import scipy.io as scio
import numpy as np


#这里给出大家注释方便理解
class MyGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=True):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['data.pt']
    # #用于从网上下载数据集
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
        ...
    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        # 这里用于构建data
        data_list = []
        for i in range(800):
            edge_index = scio.loadmat("directory of edge index")
            edge_index = torch.tensor(edge_index['edge_index'], dtype=torch.long)

            x = scio.loadmat("directory of node embeddings")
            x = torch.tensor(x['x'], dtype=torch.float)

            y = scio.loadmat("directory of multi-class label")
            y = torch.tensor(y['y'], dtype=torch.long)

            y1 = scio.loadmat("directory of binary label")
            y1 = torch.tensor(y1['y1'], dtype=torch.long)

            A = scio.loadmat("directory of adjacency matrix")
            A = torch.tensor(A['A'], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y[i], y1=y1[i], A=A)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def main():
    b = MyGraphDataset("name of graph dataset")

    print()
    print(f'Dataset: {b}:')
    print('====================')
    print(f'Number of graphs: {len(b)}')
    print(f'Number of features: {b.num_features}')
    print(f'Number of classes: {b.num_classes}')

    data = b[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print(b[0])


if __name__ == '__main__':
    main()
