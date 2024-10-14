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
        #self.H = torch.zeros(hidden_size, dtype=dtype)
        self.register_buffer('H', torch.randn(hidden_size, dtype=dtype))
    
    def reset_hidden(self):
        with torch.no_grad():
            self.H = torch.zeros_like(self.H)

    def forward(self, x):
        self.H.detach_()
        self.H = torch.sigmoid(self.A(x)) * self.H + torch.sigmoid(self.B(x)) 
        y = (self.C(torch.sigmoid(self.H))) + self.D(x)
        
        return y
