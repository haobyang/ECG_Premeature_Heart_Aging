import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size = 12, hidden_size = 128, num_layers = 1):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
            #bidirectional = True,
        )
        
        #self.out = nn.Linear(hidden_size*2, num_classes)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        
        # print('original x {}'.format(x.shape))
        # print('after changing x {}'.format(x.squeeze(dim=1).permute(0,2,1).shape))
        r_out, (h_n, h_c) = self.rnn(x.squeeze(dim=1).permute(0,2,1), None)   # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1, :])
        return out
    
