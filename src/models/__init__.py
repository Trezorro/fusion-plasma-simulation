import lightning as L

from src.models.CausalConv1d import AutoRegressiveModel
from src.models.EncoderDecoder import EncoderDecoder
from src.models.unet import UNet
from src.models.UNet_fourier import UNetFourier
from src.models.complexnet import ComplexNet


class DummyModel(L.LightningModule):

    def __init__(self, **kwargs):
        pass

    def forward(self, x, *args):
        return x

    def log_summary(self, conf):
        pass
