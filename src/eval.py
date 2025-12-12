import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import KTData


def train_eval(model, test_loader, threshold):
    total_loss = 0
    total_correct = 0
    total_predictions = 0
    total_preds, total_trues = [], []
    for train_data, response in tqdm(test_loader):
        response = torch.squeeze(response, dim=1)
        with torch.no_grad():
            predict = model(train_data, response)

        response = torch.flatten(response).float()
        predict = torch.flatten(predict)

        # 展平并过滤padding（假设padding标签为-1）
        mask = response.flatten() > -0.9  # 有效样本掩码
        valid_response = response.flatten()[mask].float()
        valid_predict = predict.flatten()[mask]

        total_preds.append(valid_predict)
        total_trues.append(valid_response)

        correct = ((valid_predict >= threshold) == (valid_response >= threshold)).float().sum()
        total_correct += correct.item()
        total_predictions += valid_response.numel()
    total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
    total_preds /= 100
    total_preds = np.clip(total_preds, 0, 1)
    total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()
    # train_accuracy = total_correct / total_predictions
    auc = roc_auc_score(y_true=total_trues, y_score=total_preds)
    acc = accuracy_score(y_true=total_trues >= 0.5, y_pred=total_preds >= 0.5)
    return auc, acc

# 测试
if __name__ == '__main__':
    num_processes = 10  # multiprocessing
    # num_processes = 1

    batch_size = 32
    dataset_name = 'assist09'
    n_problem, n_skill, n_qtype = 17751, 149, 5
    d_model = 256
    learning_rate = 0.001
    total_epoch = 300
    threshold = 0.5
    dummy = False

    test_dataset = KTData('test',
                          dataset_name,
                          dummy=dummy)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=num_processes  # multiprocessing
    )
    model = torch.load('../models/backup1/best_accuracy_model.pth')
    loss_fn = nn.BCEWithLogitsLoss()
    auc, acc = train_eval(model, test_loader, loss_fn, 0.5)
    print(f'auc: {auc}, acc:{acc}')