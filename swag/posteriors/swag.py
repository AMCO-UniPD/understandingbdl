import torch

from .subspaces import Subspace
from ._assess_dimension import _infer_dimension_

class SWAG(torch.nn.Module):

    def __init__(self, base, subspace_type,
                 subspace_kwargs=None, var_clamp=1e-6, *args, **kwargs):
        super(SWAG, self).__init__()

        # Forward architecture-specific arguments (including model_config) to base.
        self.base_model = base(*args, **kwargs)
        # Complex parameters are represented by concatenated real and imaginary
        # parts in the SWAG statistics and subspace.
        self.num_parameters = sum(param.numel() * (2 if param.is_complex() else 1)
                                  for param in self.base_model.parameters())

        self.register_buffer('mean', torch.zeros(self.num_parameters))
        self.register_buffer('sq_mean', torch.zeros(self.num_parameters))
        self.register_buffer('n_models', torch.zeros(1, dtype=torch.long))

        # Initialize subspace
        if subspace_kwargs is None:
            subspace_kwargs = dict()
        self.subspace = Subspace.create(subspace_type, num_parameters=self.num_parameters,
                                        **subspace_kwargs)

        self.var_clamp = var_clamp

        self.cov_factor = None
        self.model_device = 'cpu'

    # dont put subspace on cuda?
    def cuda(self, device=None):
        self.base_model.cuda(device=device)
        self.model_device = 'cuda'
        return self

    def cpu(self):
        return self.to('cpu')

    def to(self, *args, **kwargs):
        self.base_model.to(*args, **kwargs)
        # Use the resulting model device instead of PyTorch's private _parse_to
        # API, whose return signature differs across versions. All .to overloads
        # are handled by the base model; posterior statistics remain on CPU.
        device = next(self.base_model.parameters(),
                      next(self.base_model.buffers(), self.mean)).device
        self.model_device = device.type
        if self.subspace is not None:
            # Subspace storage remains on CPU; SWAG vectors are assembled there.
            self.subspace.to(device=torch.device('cpu'))
        return self

    @staticmethod
    def _flatten_parameter(param):
        value = param.detach().cpu().reshape(-1)
        if param.is_complex():
            return torch.cat((value.real, value.imag))
        return value

    def _assign_vector(self, vector):
        offset = 0
        with torch.no_grad():
            for param in self.base_model.parameters():
                count = param.numel()
                if param.is_complex():
                    real = vector[offset:offset + count].reshape(param.shape)
                    offset += count
                    imag = vector[offset:offset + count].reshape(param.shape)
                    offset += count
                    param.copy_(torch.complex(real, imag).to(param.device))
                else:
                    param.copy_(vector[offset:offset + count].reshape(param.shape).to(param.device))
                    offset += count

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def collect_model(self, base_model, *args, **kwargs):
        # need to refit the space after collecting a new model
        self.cov_factor = None

        w = torch.cat([self._flatten_parameter(param) for param in base_model.parameters()])
        # first moment
        self.mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.mean.add_(w / (self.n_models.item() + 1.0))

        # second moment
        self.sq_mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.sq_mean.add_(w ** 2 / (self.n_models.item() + 1.0))

        dev_vector = w - self.mean

        if self.subspace is not None:
            self.subspace.collect_vector(dev_vector, *args, **kwargs)
        self.n_models.add_(1)

    def _get_mean_and_variance(self):
        variance = torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp)
        return self.mean, variance

    def fit(self):
        if self.cov_factor is not None:
            return
        self.cov_factor = self.subspace.get_space()

    def set_swa(self):
        self._assign_vector(self.mean)

    def sample(self, scale=0.5, diag_noise=True):
        self.fit()
        mean, variance = self._get_mean_and_variance()

        eps_low_rank = torch.randn(self.cov_factor.size()[0])
        z = self.cov_factor.t() @ eps_low_rank
        if diag_noise:
            z += variance.sqrt() * torch.randn_like(variance)
        z *= scale ** 0.5
        sample = mean + z

        # apply to parameters
        self._assign_vector(sample)
        return sample

    def sample_and_assign(self, scale=0.5, use_subspace=True):
        """Draw a diagonal/low-rank SWAG sample and assign it to the model."""
        device = next(self.base_model.parameters(), self.mean).device
        mean = self.mean.to(device)
        variance = torch.clamp(self.sq_mean.to(device) - mean ** 2,
                               min=self.var_clamp)
        sampled = mean + scale * variance.sqrt() * torch.randn_like(mean)
        if use_subspace and self.subspace is not None and hasattr(self.subspace, 'cov_mat_sqrt'):
            cov_factor = self.subspace.cov_mat_sqrt.to(device)
            rank = cov_factor.shape[0]
            if rank > 0 and cov_factor.shape[1] == sampled.numel():
                sampled = sampled + scale * (cov_factor.t() @ torch.randn(
                    rank, device=device, dtype=cov_factor.dtype)) / max(rank - 1, 1) ** 0.5
        self._assign_vector(sampled)
        return sampled

    def restore_mean(self):
        """Restore the model parameters to the accumulated SWAG mean."""
        self._assign_vector(self.mean)

    def get_space(self, export_cov_factor=True):
        mean, variance = self._get_mean_and_variance()
        if not export_cov_factor:
            return mean.clone(), variance.clone()
        else:
            self.fit()
            return mean.clone(), variance.clone(), self.cov_factor.clone()

    def infer_dimension(self, update_max_rank=True, use_delta=True):
        if use_delta:
            delta = self.subspace.delta

        _, var, subspace = self.get_space()
        subspace /= (self.n_models.item() - 1) ** 0.5
        tr_sigma = var.sum()

        spectrum, _ = torch.eig(subspace @ subspace.t())
        spectrum, _ = torch.sort(spectrum[:,0], descending=True)
        
        new_max_rank, ll, _ = _infer_dimension_(spectrum.numpy(), tr_sigma.numpy(),
                                                self.n_models.item(), self.num_parameters, delta)

        if new_max_rank + 1 == self.subspace.max_rank and update_max_rank:
            self.subspace.max_rank += 1
