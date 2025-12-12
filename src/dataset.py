import os.path
import torch
from constants import *
from torch.utils.data import Dataset, DataLoader


ROW_LIST = {
    'assist09':
        ('sequence_len',
         'problem_id',
         'skill_id',
         'question_type',
         'response')
}
DATASET_PARA = {
    'assist09':{
        'data_dir': '../data/assist09',
        'feature': ['problem',
                    'skill']
    }
}


class KTData(Dataset):
    def __init__(self, data_type, dataset_name, sequence_len=200, dummy=False):
        self.data_dir = DATASET_PARA[dataset_name]['data_dir']
        self.row_list = ROW_LIST[dataset_name]
        self.filename = os.path.join(self.data_dir, f'{data_type}.txt')
        if dummy:
            self.filename = '../data/assist09/{}_dummy.txt'.format(data_type)
        self.data = self.read_data(sequence_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :-1, :], self.data[idx, -1:, :]

    def read_data(self, split_len):
        feature_padding = 0  # 特征填充值
        label_padding = -1  # 标签填充值（必须是无效标签值）

        len_row_list = len(self.row_list)
        data = [[] for _ in range(len_row_list - 1)]

        with open(self.filename, 'r') as f:
            lines = f.readlines()
        lines = [x.strip('\n') for x in lines]

        for student_idx in range(0, len(lines), len_row_list):
            student_lines = lines[student_idx: student_idx + len_row_list]
            seq_len = int(student_lines[0])  # 序列实际长度

            if seq_len < 5:  # 跳过过短序列
                continue

            # 处理每个字段
            for i in range(1, len_row_list):
                raw_values = list(map(int, student_lines[i].split(',')))
                field_name = self.row_list[i]  # 获取字段名（如'response'）

                # 对标签字段使用特殊padding
                padding_val = label_padding if field_name == 'response' else feature_padding

                if seq_len > split_len:  # 处理超长序列
                    for start in range(0, seq_len, split_len):
                        chunk = raw_values[start: start + split_len]
                        padded_chunk = chunk + [padding_val] * (split_len - len(chunk))
                        data[i - 1].append(padded_chunk)
                else:
                    padded_data = raw_values + [padding_val] * (split_len - seq_len)
                    data[i - 1].append(padded_data)

        tensor_list = [torch.tensor(field_data, dtype=torch.long) for field_data in data]
        return torch.stack(tensor_list).transpose(0, 1)  # [batch, features, seq_len]


if __name__ == '__main__':
    dataset = KTData( 'train', 'assist09')
    dataloader = DataLoader(dataset, 20)
    for data in dataloader:
        print(0)