import torch
import torch.nn as nn

# Text Generation
# input_size  = 128 => vocabulary size (128 ASCII characters)
# hidden_size = 256 => larger state needed to memorise vocabulary
#                     patterns and long-range style dependencies
# num_layers  = 3   => deeper network for richer language representations;
#                     text has more complex structure than time series
# dropout     = 0.5 => stronger regularisation since text models
#                     overfit easily on limited corpora
# output_size = 128 => one logit per vocabulary token


class TextGenLSTM(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=256, num_layers=3, dropout=0.5):
        super(TextGenLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return self.fc(out)


model = TextGenLSTM()
print(model)

x = torch.randn(16, 100, 128)
print("Input shape: ", x.shape)
print("Output shape:", model(x).shape)
