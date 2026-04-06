# Import libraries: --->
from torch.utils.data import DataLoader, random_split

# Import custom modules: --->
from dataset import WordBoundaryDataset, collate_fn
from train import training
from test import testing

# Import constants: --->
from const import  DATASET_ROOT, BATCH_SIZE

if __name__ == "__main__":
    dataset = WordBoundaryDataset(DATASET_ROOT)

    train_len = int(0.8 * len(dataset))
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

    del dataset

    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, collate_fn = collate_fn)
    test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False, collate_fn = collate_fn)

    training(train_loader)
    testing(test_loader)
