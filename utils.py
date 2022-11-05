from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.len

