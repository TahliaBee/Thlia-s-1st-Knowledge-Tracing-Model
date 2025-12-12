import json
from constants import *
import pandas as pd
import os
from collections import defaultdict
import random
from typing import Dict, List
PROCESS_MAPPING = {'assist09':
                       {"problem": "problem_id",
                        "skill": "skill_id",
                        "question_type": "answer_type"}
                }

class KTData_Processor:
    def __init__(self, src: str, tar: str, dataset_name: str, train_ratio: float):
        self.src = src
        self.tar = os.path.join(tar, dataset_name)
        self.tar_train = os.path.join(self.tar, 'train.txt')
        self.tar_test = os.path.join(self.tar, 'test.txt')
        self.tar_dist = os.path.join(self.tar, 'dist.txt')
        self.process_para = PROCESS_MAPPING[dataset_name]
        self.mapping_dir = {text: os.path.join(self.tar, '{}2id.json'.format(text)) for text in self.process_para.keys()}
        self.dataset_info_dir = os.path.join(self.tar, 'dataset_info.json')
        self.data = self._load_with_encoding()
        self.dataset_name = dataset_name
        assert 0 <= train_ratio <= 1, "训练集比例需要满足在0到1之间"
        self.train_ratio = train_ratio

    def process(self, distribution) -> None:
        if self.dataset_name == 'assist09':
            self._subprocess(distribution=distribution, mapping_dict=self.process_para)
            return

        raise ValueError('该数据集名称输入有误或不支持')

    def _subprocess(self, mapping_dict: Dict[str, str], distribution = False) -> None:
        # 先处理问题ID和知识点ID的映射
        dataset_info = {}
        # 变量映射
        mapped_names = {mapping_item: f'mapped_{mapping_col}' for mapping_item, mapping_col in mapping_dict.items()}

        student_data = defaultdict(lambda: {
            'responses': [],
            **{f'mapped_{mapping_col}': [] for mapping_col in mapping_dict.values()}
        })

        for mapping_item, mapping_col in mapping_dict.items():
            mapped_name = mapped_names[mapping_item]
            dataset_info['n_{}'.format(mapping_item)] = self._process_mapping(mapping_col,
                                                                              mapped_name,
                                                                              mapping_item)


        for _, row in self.data.iterrows():
            user_id = row['user_id']
            student_data[user_id]['responses'].append(str(int(row['correct'])))
            for mapping_item, mapping_col in mapping_dict.items():
                mapped_name = mapped_names[mapping_item]
                student_data[user_id][mapped_name].append(str(row[mapped_name]))

        # 创建路径
        if not os.path.exists(self.tar):
            os.makedirs(self.tar)

        # 记录总数
        dataset_info['train'] = dataset_info['test'] = dataset_info['max_sequence'] = 0

        # 写入输出文件
        with open(self.tar_train, 'w') as f_train, open(self.tar_test, 'w') as f_test:
            for user_id, data in student_data.items():
                rand_num = random.random()
                if rand_num < self.train_ratio:
                    f = f_train
                    dataset_info['train'] += 1
                else:
                    f = f_test
                    dataset_info['test'] += 1
                # 第一行：问题个数
                dataset_info['max_sequence'] = max(dataset_info['max_sequence'], len(data['mapped_problem_id']))
                f.write(f"{len(data['mapped_problem_id'])}\n")

                # # 第二行：问题ID（逗号分隔）
                # f.write(",".join(data['mapped_problem_id']) + "\n")
                # # 第三行：知识点
                # f.write(",".join(data['mapped_skill_id']) + "\n")

                for mapping_item, mapping_col in mapping_dict.items():
                    mapped_name = mapped_names[mapping_item]
                    f.write(",".join(data[mapped_name]) + "\n")

                # 第四行：正误（1/0，逗号分隔）
                f.write(",".join(data['responses']) + "\n")
        with open(self.dataset_info_dir, 'w') as f1:
            json.dump(dataset_info, f1, indent=4, ensure_ascii=False)
        if distribution:
            with open(self.tar_dist, 'w') as f:
                for data in student_data.values():
                    f.write(f"{len(data['mapped_problem_id'])}\n")
        print("转换完成，结果已保存到 {}".format(self.tar))
        print("数据集信息已保存到 {}".format(self.dataset_info_dir))

    def _load_with_encoding(self):
        encodings = ['utf-8', 'latin1', 'gbk', 'gb18030', 'utf-16']
        for encoding in encodings:
            try:
                return pd.read_csv(self.src, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法解码文件 {self.src}，请检查文件编码")

    def _process_mapping(self, column: str, mapped_column_name: str, dict_content_type:str) -> int:
        """
        通用的ID映射处理函数

        Args:
            column: 原始数据列名
            mapped_column_name: 映射后的列名
            key_type: 原始ID的数据类型
            value_type: 映射ID的数据类型

        Returns:
            映射字典 {原始ID: 映射ID}
        """
        # 确保原始ID是正确的类型
        self.data[column] = self.data[column].astype(str)

        # 构建映射字典
        unique_values = sorted(self.data[column].unique())
        mapping_dict = {val: str(idx) for idx, val in enumerate(unique_values, start=1)}

        # 添加映射后的列
        self.data[mapped_column_name] = self.data[column].map(mapping_dict)
        self._save_mapping_to_json(mapping_dict, dict_content_type)

        return len(mapping_dict)

    def _save_mapping_to_json(self, mapping_dict: Dict, dict_content_type: str):
        """将映射字典保存为JSON文件"""
        filepath = self.mapping_dir[dict_content_type]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(mapping_dict, f, indent=4, ensure_ascii=False)
        print(f"已保存{dict_content_type}映射文件: {filepath}")



processor = KTData_Processor('../data/ASSISTments09/skill_builder_data_corrected_collapsed_with_no_NAskill.csv',
                             DATA_DIR,
                             'assist09',
                             0.8)
processor.process(True)