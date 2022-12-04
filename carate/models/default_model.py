import torch
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="example.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


class DefaultModel(torch.nn.Module):  # TODO reuse in chembee
    """
    Base class for the Torch models. The subpackage is highly reusable in other projects

    """

    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError(
            "Please implement your forward pass according to your model archtitecture."
        )
