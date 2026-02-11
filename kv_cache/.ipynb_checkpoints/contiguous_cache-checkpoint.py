import torch

class ContiguousKVCache:
    def __init__(self, batch_size, num_heads, max_seq_len, head_dim, device):
        self.device = device
        self.k = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device)
        self.v = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device)
        self.cur_pos = 0

    def update(self, new_k, new_v):
        B, H, T, D = new_k.shape

        self.k[:, :, self.cur_pos:self.cur_pos+T, :] = new_k
        self.v[:, :, self.cur_pos:self.cur_pos+T, :] = new_v
        self.cur_pos += T

        return (
            self.k[:, :, :self.cur_pos, :],
            self.v[:, :, :self.cur_pos, :]
        )