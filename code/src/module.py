import torch


class Swish(torch.nn.Module):
    r"""Applies the element-wise function:
    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    def forward(self, input):
        return torch.sigmoid(input) * input
