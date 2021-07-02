import torch
from torchmetrics import Metric


class MWACC(Metric):
    def __init__(self, window_size: float, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.window_size = window_size

        self.add_state("tf_sequence", default=[], dist_reduce_fx="cat")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)

        self.tf_sequence += preds == target
        self.n_samples += len(preds)

    def compute(self):
        windows = self.chunkIt(self.tf_sequence)
        out = []
        for window in windows:
            out.append(sum(window) / len(window))
        return out

    def _input_format(self, y: torch.Tensor, t: torch.Tensor):
        if y.dim() == 2:
            y = y.flatten()
        if t.dim() == 2:
            t = t.flatten()

        assert y.numel() == t.numel(), "Incorrect number of samples"

        return torch.argmax(y), t

    def chunkIt(self, seq):
        avg = len(seq) / self.window_size
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last) : int(last + avg)])
            last += avg

        return out
