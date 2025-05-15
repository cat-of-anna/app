import argparse
import pickle
import os
import torch
import datetime
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append("")
from train import DisProtModel, train_main


root_path = os.getcwd()

restypes = [
    'A', 'R', 'N', 'D', 'C',
    'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
unsure_restype = 'X'
unknown_restype = 'U'


class DisTestProtDataset(Dataset):
    def __init__(self, dict_data):
        self.ids = [d['id'] for d in dict_data]
        self.sequences = [d['sequence'] for d in dict_data]
        self.residue_mapping = {'X': 20}
        self.residue_mapping.update(dict(zip(restypes, range(len(restypes)))))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.zeros(len(self.sequences[idx]), len(self.residue_mapping))
        for i, c in enumerate(self.sequences[idx]):
            if c not in restypes:
                c = 'X'
            sequence[i][self.residue_mapping[c]] = 1

        return sequence


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser('IDRs prediction')
    parser.add_argument('--config_path', default='./code/config.yaml')
    parser.add_argument('--model_save_path', type=str, default='best_model.pth')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--result_save_path')
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)

    if args.do_train:
        train_main(args)

    if args.do_predict:
        # Load test data from /saisdata，在app同级目录
        app_upper_path = os.path.abspath(os.path.join(root_path, '..'))
        saisdata_path = os.path.join(app_upper_path, 'saisdata')
        test_data_path = os.path.join(saisdata_path, os.listdir(saisdata_path)[0])

        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        test_dataset = DisTestProtDataset(test_data)

        ckpt_path = args.model_save_path
        time = ckpt_path[-16:-1]
        config_path = os.path.join(ckpt_path, 'config.yaml')
        model_path = os.path.join(ckpt_path, 'best_model.pth')
        config = OmegaConf.load(config_path)
        # _, _, test_dataset = make_dataset(config.data)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Load model
        model = DisProtModel(config.model).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Generate predictions
        columns = ['proteinID', 'sequence', 'IDRs']
        results = pd.DataFrame(columns=columns)
        with torch.no_grad():
            for i, sequences in enumerate(test_dataloader):
                sequence = sequences[0].to(device)
                pred = model(sequence.unsqueeze(0))
                pred_labels = torch.argmax(pred, dim=-1).squeeze().cpu().numpy()
                pred_str = ''.join(map(str, pred_labels.astype(int)))
                sequence_str = test_dataset.sequences[i]
                row = pd.DataFrame([{
                        columns[0]: test_dataset.ids[i],
                        columns[1]: sequence_str,
                        columns[2]: pred_str,
                    }])
                results = pd.concat([results, row], ignore_index=True)

        # Save to /saisresult
        path_is_exist = os.path.join(args.result_save_path, time)
        if not os.path.exists(path_is_exist):
            os.makedirs(path_is_exist)

        # save_path = os.path.join(app_upper_path, '/saisresult', 'submit_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv')
        save_path = os.path.join(app_upper_path, '/saisresult', 'submit.csv')
        results.to_csv(save_path, header=['proteinID', 'sequence', 'IDRs'], index=False)
        print(f'Save results to: {save_path}')
        print('Finish Inference')