import lightning as L


class DummyModel(L.LightningModule):

    def __init__(self, **kwargs):
        pass

    def forward(self, x, *args):
        return x

    def log_summary(self, conf):
        pass