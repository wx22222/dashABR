import os
import numpy as np
import torch
import torch.nn as nn
from torchviz import make_dot  # 需要先: pip install torchviz graphviz

# 和 train_lstm_fcc.py 中保持一致的超参数
hist_len = 30
hidden = 64
layers = 2
dropout = 0.2

class LSTMModel(nn.Module):
    def __init__(self, hist_len, hidden, layers, dropout):
        super().__init__()
        self.hist_len = hist_len
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=float(dropout),
        )
        self.dropout = nn.Dropout(p=float(dropout))
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, hist_len)
        x = x.unsqueeze(-1)          # (B, hist_len, 1)
        o, _ = self.lstm(x)          # (B, hist_len, hidden)
        h = o[:, -1, :]              # (B, hidden)
        h = self.dropout(h)
        y = self.fc(h)               # (B, 1)
        return y.squeeze(-1)         # (B,)

def main():
    model = LSTMModel(hist_len, hidden, layers, dropout)

    # 构造一个假输入，用于生成计算图
    dummy_input = torch.zeros(1, hist_len, dtype=torch.float32)

    # 前向计算
    output = model(dummy_input)

    # 使用 torchviz 生成计算图
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = "png"  # 输出格式: png / pdf 都可以
    os.makedirs("model_viz", exist_ok=True)
    out_path = os.path.join("model_viz", "fcc_lstm_arch")
    dot.render(out_path, cleanup=True)
    print(f"模型结构图已生成: {out_path}.png")

if __name__ == "__main__":
    main()