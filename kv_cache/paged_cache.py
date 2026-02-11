import torch

class PagedKVCache:
    """
    Page-based KV cache.

    Allocates fixed-size pages on demand.
    Avoids contiguous max_seq_len allocation.
    """
# NOTE:
# Token-by-token insertion is simple but not optimal for large T.
# For production systems, bulk page writes would be preferred.
    
    def __init__(self, batch_size, num_heads, head_dim,
                 page_size=256, device="cuda"):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.device = device

        self.k_pages = []
        self.v_pages = []
        self.cur_pos = 0  # total logical tokens stored

    def _allocate_page(self):
        k_page = torch.zeros(
            self.batch_size,
            self.num_heads,
            self.page_size,
            self.head_dim,
            device=self.device
        )
        v_page = torch.zeros_like(k_page)

        self.k_pages.append(k_page)
        self.v_pages.append(v_page)

    def update(self, new_k, new_v):
        """
        Insert new KV tensors into paged storage.
        Does NOT return concatenated KV.
        """

        B, H, T, D = new_k.shape

        for t in range(T):
            page_idx = self.cur_pos // self.page_size
            offset = self.cur_pos % self.page_size

            if page_idx >= len(self.k_pages):
                self._allocate_page()

            self.k_pages[page_idx][:, :, offset, :] = new_k[:, :, t, :]
            self.v_pages[page_idx][:, :, offset, :] = new_v[:, :, t, :]

            self.cur_pos += 1