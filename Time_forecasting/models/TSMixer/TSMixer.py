import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, configs, seq_len, enc_in):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, seq_len),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


class Model(nn.Module):
    def __init__(self, configs, seq_len, pred_len, enc_in):
        super(Model, self).__init__()
        self.task_name = "long_term_forecast"
        self.layer = configs.e_layers
        self.model = nn.ModuleList([ResBlock(configs, seq_len, enc_in)
                                    for _ in range(configs.e_layers)])
        self.pred_len = pred_len
        self.projection = nn.Linear(seq_len, pred_len)

    def forecast(self, x_enc):

        # x: [B, L, D]
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)

        return enc_out

    def forward(self, x_enc):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')
