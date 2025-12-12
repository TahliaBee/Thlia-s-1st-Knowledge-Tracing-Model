import torch
from torch.utils.data import Dataset, DataLoader
import os
import linecache
import subprocess
import sys
from typing import Union, List, Dict, overload


class KTData(Dataset):
    def __init__(self, data_dir: str, feature_index: Dict, input_features: List[str]):
        self.data_dir = data_dir
        self.lines = Lines(self.data_dir)
        self.input_features = input_features
        self.feature_index = feature_index
        '''
        {
            "user_id": 0,
            "problem_id": 1,
            "skill_id": 2,
            "correct": 3,
            "timestamp": 4
        }
        '''

    def __getitem__(self, idx: int or slice) -> Dict[str, torch.Tensor]:
        """获取单个学生的交互序列

                Returns:
                    Dict[str, Tensor]: {
                        'features': 形状 [seq_len, num_features],
                        'mask': 形状 [seq_len],  # 实际数据为1，填充为0
                        'difficulty': 形状 [seq_len]  # 题目难度标签
                    }
                """
        # 读取原始数据行
        raw_data = self.lines[idx].strip().split(',')

        # 转换为特征字典
        sample = {
            feat: int(raw_data[self.feature_index[feat]])
            for feat in self.input_features
        }

        # # 添加难度信息（如果提供）
        # if self.difficulty_levels:
        #     problem_id = int(raw_data[self.feature_index["problem_id"]])
        #     sample["difficulty"] = self.difficulty_levels.get(problem_id, -1)

        # # 数据增强
        # if self.transform:
        #     sample = self.transform(sample)

        return self._pad_sequence(sample)

    def __len__(self):
        return len(self.lines)

    def _pad_sequence(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """序列填充/截断处理"""
        seq_len = min(len(sample["correct"]), self.seq_len)

        # 特征矩阵
        features = torch.zeros((self.seq_len, len(self.input_features)), dtype=torch.long)
        features[:seq_len] = torch.tensor([
            [sample[feat][i] for feat in self.input_features]
            for i in range(seq_len)
        ])

        # 掩码（1表示真实数据，0表示填充）
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        mask[:seq_len] = 1

        # 难度标签（可选）
        difficulty = torch.full((self.seq_len,), -1, dtype=torch.long)
        if "difficulty" in sample:
            difficulty[:seq_len] = torch.tensor(sample["difficulty"][:seq_len])

        return {
            "features": features,
            "mask": mask,
            "difficulty": difficulty
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """批处理函数（用于DataLoader）"""
        return {
            "features": torch.stack([item["features"] for item in batch]),
            "masks": torch.stack([item["mask"] for item in batch]),
            "difficulties": torch.stack([item["difficulty"] for item in batch])
        }


class Lines:
    def __init__(self, filename: str, skip: int = 0, group: int = 1, preserve_newline: bool = False):
        """高效随机访问大文本文件的类

        Args:
            filename: 文件路径
            skip: 跳过的初始行数
            group: 每组包含的行数
            preserve_newline: 是否保留换行符
        """
        self.filename = filename
        # 快速验证文件可访问性
        with open(filename) as f:
            pass

            # 跨平台行数统计
        self.linecount = self._count_lines()
        self.length = (self.linecount - skip) // group
        self.skip = skip
        self.group = group
        self.preserve_newline = preserve_newline

    def _count_lines(self) -> int:
        """跨平台高效统计文件行数"""
        if sys.platform == "win32":
            with open(self.filename, 'rb') as f:
                return sum(1 for _ in f)
        else:
            output = subprocess.check_output(['wc', '-l', self.filename])
            return int(output.split()[0])

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> 'LinesIterator':
        return LinesIterator(self)

    @overload
    def __getitem__(self, item: int) -> Union[str, List[str]]:
        ...

    @overload
    def __getitem__(self, item: slice) -> List[Union[str, List[str]]]:
        ...

    def __getitem__(self, item: Union[int, slice]) -> Union[str, List[str], List[Union[str, List[str]]]]:
        """获取单行、行组或切片

        Args:
            item: 整数索引或切片对象

        Returns:
            单行字符串(group=1) / 行组列表(group>1) / 切片结果列表
        """
        if isinstance(item, int):
            return self._get_single_item(item)
        elif isinstance(item, slice):
            return self._get_slice(item)
        raise TypeError(f"Invalid index type: {type(item)}")

    def _get_single_item(self, index: int) -> Union[str, List[str]]:
        """获取单个索引对应的行/行组"""
        if not -len(self) <= index < len(self):
            raise IndexError("Index out of range")

        if index < 0:
            index += len(self)

        line_num = self.skip + 1 + index * self.group
        if self.group == 1:
            return self._process_line(linecache.getline(self.filename, line_num))
        return [
            self._process_line(linecache.getline(self.filename, line_num + k))
            for k in range(self.group)
        ]

    def _get_slice(self, sl: slice) -> List[Union[str, List[str]]]:
        """处理切片访问"""
        start, stop, _ = sl.indices(len(self))
        return [self._get_single_item(i) for i in range(start, stop)]

    def _process_line(self, line: str) -> str:
        """处理单行文本（去除换行符等）"""
        return line if self.preserve_newline else line.rstrip('\r\n')


class LinesIterator:
    """迭代器实现（分离迭代逻辑）"""

    def __init__(self, lines: Lines):
        self.lines = lines
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.lines):
            raise StopIteration
        result = self.lines[self.index]
        self.index += 1
        return result


def _clip(value: int, min_val: int, max_val: int) -> int:
    """限制数值在[min_val, max_val]范围内"""
    return max(min(value, max_val), min_val)

if __name__ == '__main__':
    dataset = KTData("../data/assist09_from DTransformer/train.txt", {'b': 1, 'c': 2, 'd': 3}, ['b', 'c', 'd'])
    dataloader = DataLoader(dataset, 8)
    for data in dataloader:
        print(0)
