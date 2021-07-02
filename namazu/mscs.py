import torch
from torchmetrics import Metric


class RMSCS(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        self.add_state("distance", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, centre: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        # assert preds.shape == target.shape

        distance = self.cos(preds, centre)
        self.distance += torch.sum(torch.square(distance))
        self.n_samples += len(preds)

    def compute(self):

        return torch.sqrt(self.distance.float() / self.n_samples)

    def _input_format(self):
        pass
