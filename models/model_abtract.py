from abc import ABC, abstractmethod
import torch
import numpy as np
from sklearn.metrics import top_k_accuracy_score


class BaseModel(ABC):
    """
    Abstract base class for all models
    """

    # @abstractmethod
    # def encode(self):
    #     ...

    # Validation
    validation_step_even_emb = []
    validation_step_uneven_emb = []
    validation_step_idx = []

    @abstractmethod
    def load_model(self):
        ...

    @property
    def name(self):
        ...

    def val_step(self, batch, batch_idx):
        even, uneven, idx = batch
        even_emb = self.predict_step(even, batch_idx)
        uneven_emb = self.predict_step(uneven, batch_idx)
        self.validation_step_even_emb.append(even_emb)
        self.validation_step_uneven_emb.append(uneven_emb)
        self.validation_step_idx.append(torch.tensor(idx))
        return (even_emb, uneven_emb, idx)
    
    def on_val_end(self):
        even_emb = torch.cat(self.validation_step_even_emb, dim=0)#.detach().cpu()
        uneven_emb = torch.cat(self.validation_step_uneven_emb, dim=0)#.detach().cpu()
        idx = torch.cat(self.validation_step_idx, dim=0)#.detach().cpu()

        #sims = (even_emb[:, None, :] * uneven_emb[None, :, :]).sum(dim=2)

        batch_size = 256
        sims = []
        for i in range(0, len(even_emb), batch_size):
            even_batch = even_emb[i:i+batch_size]
            sim = (even_batch[:, None, :] * uneven_emb[None, :, :]).sum(dim=2).detach().cpu()
            sims.append(sim)
        sims = torch.cat(sims, dim=0)

        sims = sims.squeeze()
        labels = np.arange(sims.shape[1])
        res = top_k_accuracy_score(y_true=idx, y_score=sims, labels=labels, k=1, normalize=True)
        self.validation_step_even_emb.clear()
        self.validation_step_uneven_emb.clear()
        self.validation_step_idx.clear()
        return res
