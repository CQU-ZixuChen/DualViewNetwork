import torch
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])
if torch.cuda.is_available():
    args.device = 'cuda:0'


# loss function
def cal_loss(model, data1):
    data1 = data1.to(args.device)
    output0, output1, feature0, feature1 = model(data1, 1)
    output2 = torch.cat((output1[:, 0].unsqueeze(1), torch.sum(output1[:, 1:3], dim=1).unsqueeze(1)), 1)
    weight = 2 * torch.sigmoid(1 - F.cosine_similarity(output0, output2, dim=1))

    y0 = F.one_hot(data1.y1, num_classes=2)
    loss0 = - y0 * torch.log(output0 + 1e-6)
    loss0 = torch.sum(loss0) / y0.shape[0]

    y1 = F.one_hot(data1.y, num_classes=4)
    loss1 = - y1 * torch.log(output1 + 1e-6)
    loss1 = torch.sum(weight * torch.sum(loss1, dim=1)) / y1.shape[0]

    loss = 0.5 * loss0 + 0.5 * loss1

    return loss

# test function
def tst(model, loader):
    model.eval()
    correct = 0.
    for data in loader:
        data = data.to(args.device)
        output0, output1, feature0, feature1 = model(data, 0)
        output2 = torch.cat((output1[:, 0].unsqueeze(1), torch.sum(output1[:, 1:3], dim=1).unsqueeze(1)), 1)
        weight = 2 * torch.sigmoid(1 - F.cosine_similarity(output0, output2, dim=1))

        y0 = F.one_hot(data.y1, num_classes=2)
        loss0 = - y0 * torch.log(output0 + 1e-6)
        loss0 = torch.sum(loss0) / y0.shape[0]

        y1 = F.one_hot(data.y, num_classes=4)
        loss1 = - y1 * torch.log(output1 + 1e-6)
        loss1 = torch.sum(weight * torch.sum(loss1, dim=1)) / y1.shape[0]

        loss = 0.5 * loss0 + 0.5 * loss1

        output1 = output1.max(dim=1)[1]
        correct += output1.eq(data.y).sum().item()
        accuracy = correct / len(data.y)

    return loss, accuracy

