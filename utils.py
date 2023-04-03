import torch

def get_accuracy(model, data):
    loader = torch.utils.data.DataLoader(data, batch_size=100)
    correct, total = 0, 0
    for xs, ts in loader:
        pred = model(xs)
        correct += torch.sum(torch.argmax(pred, 1) == torch.argmax(ts, 1)).item()
        total += int(ts.shape[0])
        return correct / total