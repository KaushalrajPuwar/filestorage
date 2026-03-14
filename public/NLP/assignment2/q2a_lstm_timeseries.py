import torch
import torch.nn as nn

# Time Series Prediction
# input_size  = 1   => one value per timestep (univariate series)
# hidden_size = 64  => enough capacity to learn temporal patterns
#                     without overfitting on small datasets
# num_layers  = 2   => captures both short and medium-range dependencies
# dropout     = 0.2 => light regularisation between layers to
#                     prevent overfitting on noisy time series
# output_size = 1   => predicting a single future value

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return self.fc(out[:, -1, :])


model = TimeSeriesLSTM()
print(model)

x = torch.randn(32, 50, 1)
print("Input shape: ", x.shape)
print("Output shape:", model(x).shape)
