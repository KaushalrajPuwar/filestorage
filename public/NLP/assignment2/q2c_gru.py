import torch
import torch.nn as nn

class BasicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BasicGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Update gate weights
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        # Reset gate weights
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        # Candidate hidden state weights
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat([x_t, h], dim=1)

            # Update gate
            z_t = torch.sigmoid(self.W_z(combined))
            # Reset gate
            r_t = torch.sigmoid(self.W_r(combined))
            # Candidate hidden state
            h_candidate = torch.tanh(self.W_h(torch.cat([x_t, r_t * h], dim=1)))
            # New hidden state
            h = (1 - z_t) * h + z_t * h_candidate

            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1)


input_size  = 10
hidden_size = 50
num_layers  = 2

gru_model = BasicGRU(input_size, hidden_size, num_layers)
print(gru_model)

x = torch.randn(32, 20, input_size)
print("Input shape: ", x.shape)
print("Output shape:", gru_model(x).shape)

print("\n------Cross-check with nn.GRU------\n")
# Cross-check: output shape should match nn.GRU
ref = nn.GRU(input_size, hidden_size, batch_first=True)
ref_out, _ = ref(x)
print("nn.GRU output shape:", ref_out.shape)
print("Shapes match:", gru_model(x).shape == ref_out.shape)

# Check hidden state is actually changing across timesteps
out = gru_model(x)
print("Are gates Broken? => Timestep 0 == Timestep 1:", torch.allclose(out[:, 0, :], out[:, 1, :]))
# Should print False if working and True if the gates are broken
