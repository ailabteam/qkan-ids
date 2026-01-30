import torch
import torch.nn as nn

class QKAN(nn.Module):
    def __init__(self, layers_hidden, num_qlayers=1, device='cuda'):
        super(QKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(QKANLayer(layers_hidden[i], layers_hidden[i+1], num_qlayers, device))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class QKANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_qlayers, device):
        super(QKANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        
        # Tham số cho Quantum-inspired activation (Eq 4, 5)
        self.alpha = nn.Parameter(torch.randn(in_dim, out_dim, device=device))
        self.beta = nn.Parameter(torch.randn(in_dim, out_dim, device=device))
        self.omega = nn.Parameter(torch.randn(in_dim, out_dim, 3, device=device))

    def forward(self, x):
        # x: [batch, in_dim]
        x_ext = x.unsqueeze(2).expand(-1, -1, self.out_dim) # [batch, in_dim, out_dim]
        theta = self.alpha * x_ext + self.beta
        
        # Mô phỏng mạch Quantum PQC (Fourier Series)
        phi = torch.cos(theta + self.omega[:,:,0]) * torch.cos(theta * self.omega[:,:,1] + self.omega[:,:,2])
        return torch.sum(phi, dim=1)
