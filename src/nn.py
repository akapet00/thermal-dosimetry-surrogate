import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import trange

from .module import Swish


class AdaptiveLinear(torch.nn.Linear):
    r"""Applies a linear transformation to the input data as
    
    .. math::
        y = naxA^T + b
        
    More details available in Jagtap, A. D. et al. Locally adaptive
    activation functions with slope recovery for deep and
    physics-informed neural networks, Proc. R. Soc. 2020.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 adaptive_rate=None,
                 adaptive_rate_scaler=None):
        """Constructor.
        
        Parameters
        ----------
        in_features : int
            Size of each input sample.
        out_features : int
            Size of each output sample.
        bias : bool, optional
            If set to `False`, a layer will not learn an additive bias.
        adaptive_rate : float, optional
            Scalable adaptive rate parameter for activation function
            that is added layer-wise for each neuron separately. It is
            treated as learnable parameter and will be optimized using
            an optimizer of choice.
        adaptive_rate_scaler : float, optional
            Fixed, pre-defined, scaling factor for adaptive activation
            functions.
            
        Returns
        -------
        None
        """
        super(AdaptiveLinear, self).__init__(in_features, out_features, bias)
        self.adaptive_rate = adaptive_rate
        self.adaptive_rate_scaler = adaptive_rate_scaler
        if self.adaptive_rate:
            self.A = torch.nn.Parameter(
                self.adaptive_rate * torch.ones(self.in_features)
            )
            if not self.adaptive_rate_scaler:
                self.adaptive_rate_scaler = 10.0

    def forward(self, input):
        if self.adaptive_rate:
            return torch.nn.functional.linear(
                self.adaptive_rate_scaler * self.A * input,
                self.weight,
                self.bias
            )
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}, '
            f'adaptive_rate={self.adaptive_rate is not None}, '
            f'adaptive_rate_scaler={self.adaptive_rate_scaler is not None}'
        )


class MLP(torch.nn.Module):
    r"""Multilayer perceptron module."""
    def __init__(self,
                 sizes,
                 activation,
                 dropout=0.0,
                 adaptive_rate=None,
                 adaptive_rate_scaler=None):
        """Constructor.
        
        Parameters
        ----------
        sizes : list
            Number of hidden units per hidden layer.
        activation : str
            Activation function of choice
        dropout : float, optional
            If set to float between 0 and 1, the dropout regularization
            will be applied.
        adaptive_rate : float, optional
            Scalable adaptive rate parameter for activation function
            that is added layer-wise for each neuron separately. It is
            treated as learnable parameter and will be optimized using
            an optimizer of choice.
        adaptive_rate_scaler : float, optional
            Fixed, pre-defined, scaling factor for adaptive activation
            functions.
            
        Returns
        -------
        None
        """
        super(MLP, self).__init__()
        self.module = torch.nn.Sequential(
            *[MLP.linear_block(inf_f,
                               out_f,
                               activation,
                               dropout,
                               adaptive_rate,
                               adaptive_rate_scaler)
              for inf_f, out_f in zip(sizes[:-1], sizes[1:-1])],
            AdaptiveLinear(sizes[-2], sizes[-1])
        )
    
    @staticmethod
    def linear_block(inf_f, out_f, activation, dropout, adaptive_rate,
                     adaptive_rate_scaler):
        activation_dispatcher = torch.nn.ModuleDict([
            ['lrelu', torch.nn.LeakyReLU()],
            ['relu', torch.nn.ReLU()],
            ['tanh', torch.nn.Tanh()],
            ['sigmoid', torch.nn.Sigmoid()],
            ['swish', Swish()],
        ])
        return torch.nn.Sequential(
            AdaptiveLinear(inf_f,
                           out_f,
                           adaptive_rate=adaptive_rate,
                           adaptive_rate_scaler=adaptive_rate_scaler),
            activation_dispatcher[activation],
            torch.nn.Dropout(dropout)
        )

    def _get_data(self, train_ds, valid_ds, batch_size):
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(valid_ds, batch_size=batch_size*2))

    def fit(self,
            X_train,
            y_train,
            X_valid,
            y_valid,
            optimizer,
            criterion,
            iterations=100,
            batch_size=32,
            apply_slope_recovery=False):
        """Constructor.
        
        Parameters
        ----------
        X_train : torch.Tensor
            Training set features.
        y_train : torch.Tensor
            Training set targets.
        X_valid : torch.Tensor
            Validation set features.
        y_valid : torch.Tensor
            Validation set targets.
        optimizer : torch.optim.Optimizer
            Optimizer.
        criterion : torch.nn.modules.loss.Module
            Criterion for learning.
        iterations : int, optional
            Number of training epochs.
        batch_size : int, optional
            Batch size.
        apply_slope_recovery : bool, optional
            Apply slope recovery term to the loss function. `False` by
            default.
        
        Returns
        -------
        tuple
            Two lists containing train and validation loss values.
        """
        train_ds = TensorDataset(X_train, y_train)
        valid_ds = TensorDataset(X_valid, y_valid)
        train_dl, valid_dl = self._get_data(train_ds, valid_ds, batch_size)
        train_loss = []
        valid_loss = []
        for _ in trange(iterations, desc='Training', total=iterations):
            loss_history = []
            self.train()
            for xb_train, yb_train in train_dl:
                y_pred = self.forward(xb_train)
                if apply_slope_recovery:
                    local_recovery = torch.tensor(
                        [torch.mean(self.regressor[layer][0].A.data)
                         for layer in range(len(self.regressor) - 1)]
                    )
                    slope_recovery = 1 / torch.mean(torch.exp(local_recovery))
                    loss = criterion(y_pred, yb_train) + slope_recovery
                else:
                    loss = criterion(y_pred, yb_train)
                loss_history.append(loss.detach().numpy())
                # backprop here
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            train_loss.append(sum(loss_history) / len(loss_history))
            self.eval()
            with torch.no_grad():
                valid_loss.append(sum(
                    criterion(self.forward(xb_valid), yb_valid)
                    for xb_valid, yb_valid in valid_dl
                ).detach().numpy() / len(valid_dl))
        return train_loss, valid_loss

    def forward(self, X):
        return self.module(X)
