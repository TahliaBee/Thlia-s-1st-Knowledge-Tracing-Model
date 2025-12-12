def read_data(self, split_len):
    data = []
    with open(self.filename, 'r') as f:
        lines = f.readlines()
    len_row_list = len(self.row_list)


    for student_idx in range(0, len(lines), len_row_list):
        student_lines = lines[student_idx: student_idx + len_row_list]
        student_lines = list(map(lambda x: x.strip('\n'), student_lines))
        seq_len = student_lines[0] = int(student_lines[0])
        for i in range(1, len_row_list):
            student_lines[i] = list(map(int, student_lines[i].split(',')))
        all_student_data = []

        # 处理超长序列
        if seq_len > split_len:
            for start in range(0, seq_len, split_len):
                split_data = []
                oversize_flag = start + split_len < seq_len
                split_data.append(split_len if oversize_flag else seq_len - start)
                for i in range(1, len_row_list):
                    split_data.append(student_lines[i][start: (start + split_len) if oversize_flag else seq_len])
                all_student_data.append(split_data)
        else:
            all_student_data.append(student_lines)

        # 填充序列
        for student_data in all_student_data:
            student_data_dict = {}
            for i in range(1, len_row_list):
                student_data[i] = student_data[i] + [-1] * (split_len - student_data[0])
            for line_idx, line in enumerate(student_data):
                student_data_dict[self.row_list[line_idx]] = line
            data.append(student_data_dict)


    student_data = {}
    for i in range(len(lines)):
        if isinstance(lines[i], str) and ',' in lines[i]:
            lines[i] = lines[i].strip('\n')
            lines[i] = list(map(int, lines[i].split(',')))
            if len(lines[i]) > split_len:
                lines[i] = lines[i][:split_len]
            else:
                lines[i] += [-1] * (split_len - len(lines[i]))
        else:
            lines[i] = lines[i].strip('\n')
            lines[i] = int(lines[i])
            if lines[i] > split_len:
                lines[i] = split_len
        student_data[self.row_list[i%len_row_list]] = lines[i]
        if i > 0 and (i+1) % len_row_list == 0:
            data.append(student_data)
            student_data = {}