import torch
from torchmetrics import Metric


class ACC(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self) -> float:
        # compute final result
        return self.correct.float() / self.total

    def _input_format(self, y: torch.Tensor, t: torch.Tensor):
        if y.dim() == 2:
            y = y.flatten()
        if t.dim() == 2:
            t = t.flatten()

        assert y.numel() == t.numel(), "Incorrect number of samples"

        return torch.argmax(y), t
