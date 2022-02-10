import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SimCLRLoss', 'simclr_loss']

def simclr_loss(out1, out2, temperature, normalize=False):
    """
    Compute NT_xent Loss
    """
    assert out1.size(0) == out2.size(0)
    if normalize:
        out1 = F.normalize(out1)
        out2 = F.normalize(out2)
    N = out1.size(0)

    _out = [out1, out2]
    outputs = torch.cat(_out, dim=0)

    sim_matrix = outputs @ outputs.t()
    sim_matrix = sim_matrix / temperature

    sim_matrix.fill_diagonal_(-5e4)
    sim_matrix = F.log_softmax(sim_matrix, dim=1)
    loss = -torch.sum(sim_matrix[:N, N:].diag() + sim_matrix[N:, :N].diag()) / (2*N)

    return loss

class SimCLRLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(SimCLRLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples()
    
    def mask_correlated_samples(self):
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0

        return mask
        
    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), 
        we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size 

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
