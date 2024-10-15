import torch
import torch.nn as nn

#Network implementation
class SSM(nn.Module):
    def __init__(self, sequence_size, hidden_size, output_size, dtype = torch.float32):
        super(SSM, self).__init__()
        self.A = nn.Linear(sequence_size, hidden_size, dtype= dtype)
        self.B = nn.Linear(sequence_size, hidden_size, dtype= dtype)
        self.C = nn.Linear(hidden_size, output_size, dtype= dtype)
        self.D = nn.Linear(sequence_size, output_size, dtype= dtype)
        self.H = torch.randn(hidden_size, dtype=dtype)
        self.hidden_size = hidden_size
        self.sequence_size = sequence_size
        self.output_size = output_size
        #self.register_buffer('H', None)
    
    def reset_hidden(self):
        #self.H = torch.zeros(batch_size, self.hidden_size, dtype=self.A.weight.dtype, device=self.A.weight.device)
        self.H = torch.zeros(self.hidden_size, dtype = self.A.weight.dtype)

    def forward(self, x):
        self.H.detach_()
        b = self.B(x)
        self.H = torch.tanh(self.A(x)) * self.H + torch.sigmoid(b) * b 
        y = (self.C(torch.tanh(self.H))) + self.D(x)
        
        return y
