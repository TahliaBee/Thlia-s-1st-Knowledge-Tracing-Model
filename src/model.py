import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MyModel(nn.Module):
    def __init__(self, n_problem, n_skill, n_qtype, d_model, n_head=8, d_ff=1024):
        super().__init__()
        self.n_problem = n_problem
        self.n_skill = n_skill
        self.n_qtype = n_qtype
        self.embed_len = d_model

        self.com_problem_embedding = nn.Embedding(n_problem * 2 + 1, d_model, padding_idx=0)
        self.com_skill_embedding = nn.Embedding(n_skill * 2 + 1, d_model, padding_idx=0)
        self.com_qtype_embedding = nn.Embedding(n_qtype * 2 + 1, d_model, padding_idx=0)

        self.problem_embedding = nn.Embedding(n_problem + 1, d_model, padding_idx=0)
        self.skill_embedding = nn.Embedding(n_skill + 1, d_model, padding_idx=0)
        self.qtype_embedding = nn.Embedding(n_qtype + 1, d_model, padding_idx=0)

        self.problem_encoder = MultiHeadAttention(d_model, n_head)
        self.skill_encoder = MultiHeadAttention(d_model, n_head)
        self.qtype_encoder = MultiHeadAttention(d_model, n_head)

        self.problem_forward = FeedForward(d_model, d_ff)
        self.skill_forward = FeedForward(d_model, d_ff)
        self.qtype_forward = FeedForward(d_model, d_ff)

        self.problem_decoder1 = MultiHeadAttention(d_model, n_head)
        self.skill_decoder1 = MultiHeadAttention(d_model, n_head)
        self.qtype_decoder1 = MultiHeadAttention(d_model, n_head)

        self.problem_decoder2 = MultiHeadAttention(d_model, n_head)
        self.skill_decoder2 = MultiHeadAttention(d_model, n_head)
        self.qtype_decoder2 = MultiHeadAttention(d_model, n_head)

        self.predict = FeedForward(d_model, d_ff, predict=True)
        self._init_weights()

    def forward(self, x_feature, x_response):
        '''
        :param input: [bs, len(ROW_LIST[i]) - 1, seq_len]
        :return:
        '''
        # 处理n种特征
        x_problem, x_skill, x_qtype = torch.unbind(x_feature, dim=-2)
        batch_size, seq_len = x_problem.size(0), x_problem.size(1)

        x_emb_problem = self.problem_embedding(x_problem.long())
        x_emb_skill = self.skill_embedding(x_skill.long())
        x_emb_qtype = self.qtype_embedding(x_qtype.long())

        mask1 = torch.tril(torch.ones(seq_len, seq_len))
        mask2 = torch.tril(torch.ones(seq_len, seq_len), diagonal=-1)

        x_p_flow = self.problem_encoder(x_emb_problem, x_emb_problem, x_emb_problem, mask1)
        x_s_flow = self.skill_encoder(x_emb_skill, x_emb_skill, x_emb_skill, mask1)
        x_q_flow = self.qtype_encoder(x_emb_qtype, x_emb_qtype, x_emb_qtype, mask1)

        x_p_flow = self.problem_forward(x_p_flow)
        x_s_flow = self.skill_forward(x_s_flow)
        x_q_flow = self.qtype_forward(x_q_flow)

        y_problem = x_problem + x_problem * x_response
        y_skill = x_skill + x_skill * x_response
        y_qtype = x_qtype + x_qtype * x_response

        y_problem = y_problem.long()
        y_skill = y_skill.long()
        y_qtype = y_qtype.long()

        y_emb_problem = self.com_problem_embedding(y_problem)
        y_emb_skill = self.com_skill_embedding(y_skill)
        y_emb_qtype = self.com_qtype_embedding(y_qtype)

        y_p_flow = self.problem_decoder1(y_emb_problem, y_emb_problem, y_emb_problem, mask1)
        y_s_flow = self.skill_decoder1(y_emb_skill, y_emb_skill, y_emb_skill, mask1)
        y_q_flow = self.qtype_decoder1(y_emb_qtype, y_emb_qtype, y_emb_qtype, mask1)

        p_flow = self.problem_decoder2(x_p_flow, x_p_flow, y_p_flow, mask2)
        s_flow = self.skill_decoder2(x_s_flow, x_s_flow, y_s_flow, mask2)
        q_flow = self.qtype_decoder2(x_q_flow, x_q_flow, y_q_flow, mask2)

        flow = p_flow + s_flow + q_flow

        prediction = self.predict(flow)
        prediction = torch.squeeze(prediction, dim=-1)

        return prediction

    def _init_weights(self):
        """最终优化版初始化函数"""
        d_model = self.embed_len

        # 1. Embedding层（保持稳定）
        for emb in [self.com_problem_embedding, self.com_skill_embedding, self.com_qtype_embedding,
                    self.problem_embedding, self.skill_embedding, self.qtype_embedding]:
            nn.init.uniform_(emb.weight, -0.1, 0.1)
            if emb.padding_idx is not None:
                nn.init.zeros_(emb.weight[emb.padding_idx])

        # 2. 注意力层（微调增益）
        for attn in [self.problem_encoder, self.skill_encoder, self.qtype_encoder,
                     self.problem_decoder1, self.skill_decoder1, self.qtype_decoder1,
                     self.problem_decoder2, self.skill_decoder2, self.qtype_decoder2]:
            nn.init.xavier_uniform_(attn.W_Q.weight, gain=0.9)  # 调整增益
            nn.init.zeros_(attn.W_Q.bias)
            nn.init.xavier_uniform_(attn.W_K.weight, gain=1.0)  # K保持原增益
            nn.init.zeros_(attn.W_K.bias)
            nn.init.xavier_uniform_(attn.W_V.weight, gain=1.0)
            nn.init.zeros_(attn.W_V.bias)
            nn.init.xavier_uniform_(attn.out_proj.weight, gain=0.9)
            nn.init.zeros_(attn.out_proj.bias)
            nn.init.uniform_(attn.gammas, 0.98, 1.02)  # 更严格的gamma范围

        # 3. 前馈网络（扩大第二层初始化）
        for ff in [self.problem_forward, self.skill_forward, self.qtype_forward]:
            nn.init.kaiming_normal_(ff.linear1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(ff.linear1.bias)
            nn.init.xavier_uniform_(ff.linear2.weight, gain=1.5)  # 增大增益
            nn.init.zeros_(ff.linear2.bias)

        # 4. 预测层（收紧偏置初始化）
        if hasattr(self.predict, 'predict'):
            nn.init.normal_(self.predict.linear1.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.predict.linear1.bias, mean=0.0, std=0.01)  # 缩小标准差
            nn.init.normal_(self.predict.linear2.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.predict.linear2.bias, mean=0.0, std=0.01)
            nn.init.normal_(self.predict.predict.weight, mean=0.0, std=0.1)
            nn.init.constant_(self.predict.predict.bias, -2.0)

        # 5. 打印初始化统计（调试用）
        if True:  # 调试时设为True
            total_params = 0
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
                    total_params += param.numel()
            print(f"Total trainable params: {total_params / 1e6:.2f}M")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, gamma_init=0.01, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # 为每个头分配独立的gamma（可训练参数）
        self.gammas = nn.Parameter((torch.randn(n_heads, 1, 1) + 1) * gamma_init)  # [n_heads, 1, 1]

        # 线性投影层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # 定义Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, mask=None):
        bs, seq_len = Q.size(0), Q.size(1)

        # 1. 线性投影并分头
        q = self.W_Q(Q).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [bs, n_heads, seq_len, head_dim]
        k = self.W_K(K).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(V).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 2. 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [bs, n_heads, seq_len, seq_len]

        # 3. 生成每头独立的衰减权重
        positions = torch.arange(seq_len, device=Q.device)
        distance = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))  # [seq_len, seq_len]
        decay = torch.exp(-self.gammas * distance.unsqueeze(0))  # [n_heads, seq_len, seq_len]
        decay = decay.unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
        scores = scores * decay  # 乘法注入衰减

        # 4. 处理Mask和Softmax
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e4)
        attn_weights = F.softmax(scores, dim=-1)

        # 5. 对注意力权重应用Dropout
        attn_weights = self.attn_dropout(attn_weights)

        # 6. 加权求和并合并多头
        output = torch.matmul(attn_weights, v)  # [bs, n_heads, seq_len, head_dim]
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        output = self.out_proj(output)

        # 7. 对最终输出应用Dropout
        output = self.out_dropout(output)

        # 残差连接 + 层归一化
        return self.norm(Q + output)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, predict:bool=False, dropout=0.1, activation="relu"):
        super().__init__()
        self.predict = predict
        self.activation_method = activation
        self.linear1 = nn.Linear(d_model, d_ff)  # 扩展维度
        self.activation1 = self.activation_init()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # 收缩维度
        if predict:
            self.predict = nn.Linear(d_model, 1)
        self.activation2 = self.activation_init()
        self.dropout2 = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def activation_init(self):
        a = self.activation_method
        if a == 'relu':
            return nn.ReLU()
        elif a == 'gelu':
            return  nn.GELU()
        elif a == 'softplus':
            return nn.Softplus()
        else:
            raise ValueError(f"Unsupported activation: {a}")

    def forward(self, x):
        """
        输入: x [batch_size, seq_len, d_model]
        输出: [batch_size, seq_len, d_model]
        """
        output = self.dropout2(
                self.activation2(
                    self.linear2(
                        self.dropout1(
                            self.activation1(
                                self.linear1(x))))))
        if self.predict:
            return self.predict(output)
        else:
            return output


if __name__ == '__main__':
    # 测试MyModel
    n_problem, n_skill, embed_len, bs, seq_len, split_len, n_qtype = 20, 30, 32, 24, 10, 15, 5
    model = MyModel(n_problem, n_skill, n_qtype, embed_len)
    print(0)

    # x_problem = torch.randint(1, n_problem + 1, (bs, seq_len))
    # x_padding_problem = torch.cat((x_problem, torch.zeros((bs, split_len - seq_len))), -1)
    #
    # x_skill = torch.randint(1, n_skill + 1, (bs, seq_len))
    # x_padding_skill = torch.cat((x_skill, torch.zeros((bs, split_len - seq_len))), -1)
    #
    # x_response = torch.randint(0, 2, (bs, seq_len))
    # x_padding_response = torch.cat((x_response, torch.zeros((bs, split_len - seq_len))), -1)
    #
    # x_qtype = torch.randint(1, n_qtype + 1, (bs, seq_len))
    # x_padding_qtype = torch.cat((x_qtype, torch.zeros((bs, split_len - seq_len))), -1)
    #
    # x = torch.stack(
    #     (x_padding_problem, x_padding_skill, x_padding_qtype, x_padding_response), -2)
    # # 模拟联合嵌入
    #
    # x_emb_combine = model(x)

    # # 测试attention
    # gamma = torch.rand((1))
    # dropout_rate = 0.1
    # check1 = attention(x_emb_combine, x_emb_combine, x_emb_combine, embed_len, gamma, dropout_rate)