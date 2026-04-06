# Import libraries: --->
import torch as tc
from torch.utils.data import DataLoader, random_split

# Import custom modules: --->
from dataset import WordBoundaryDataset, collate_fn
from model import BiRNN
from ensemble_eval import evaluate_attention

# Import constants: --->
from const import DATASET_ROOT, BATCH_SIZE

def load_model(path, model):
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    model = BiRNN(model).to(device)

    model.load_state_dict(tc.load(path))

    return model

if __name__ == "__main__":
    # Set the model paths: --->
    model_paths = [
        ("checkpoints/wbd_ass_nt_dft_bilstm_1.pth", "BiLSTM"),            # Bi-LSTM + DFT
        # ("checkpoints/wbd_ass_nt_dct_bilstm_1.pth", "BiLSTM")           # Bi-LSTM + DCT
        ("checkpoints/wbd_ass_nt_dft_bigru_1.pth", "BiGRU")               # Bi-GRU  + DFT
        # ("checkpoints/wbd_ass_nt_dct_bigru_1.pth", "BiGRU")              # Bi-GRU  + DCT
    ]

    # Load the independent models from the defined path: --->
    # models = [load_model(path) for path in model_paths]
    models = []

    for model_path in model_paths:
        models.append(load_model(model_path[0], model_path[1]))

    # Load test dataset: --->
    test_dataset = WordBoundaryDataset(DATASET_ROOT)

    train_len = int(0.8 * len(test_dataset))
    train_set, test_set = random_split(test_dataset, [train_len, len(test_dataset) - train_len])

    del test_dataset, train_set

    test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False, collate_fn = collate_fn)

    # Run evaluation: --->
    evaluate_attention(models, test_loader)
