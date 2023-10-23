import torch
import torch.nn as nn
from random import sample

class MoCo(nn.Module):

    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=32, r = 16348, m=0.999, T=0.1):
        super(MoCo, self).__init__()

        self.r = r
        self.m = m 
        self.T = T

        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False 

        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        # pointer for enqueue and dequeue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder 

        """

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        assert self.r % batch_size == 0

        self.queue[:,ptr:ptr+batch_size] = keys.T 
        ptr = (ptr + batch_size) % self.r # move pointer 

        self.queue_ptr[0] = ptr 

    


