from torch import nn

ACTIVATION_OPTIONS = dict(
    ReLU=nn.ReLU,
    GELU=nn.GELU,  # used in many transformers
    SiLU=nn.SiLU,  # coind in same paper as GELU, slightly simpler.
    Swish=nn.SiLU,
    ELU=nn.ELU,
    SELU=nn.SELU,  # Scaled EL, self normalizing
    LeakyReLU=nn.LeakyReLU,
    Identity=nn.Identity,
    Tanh=nn.Tanh,
    Softplus=nn.Softplus,
    Sigmoid=nn.Sigmoid,
)
