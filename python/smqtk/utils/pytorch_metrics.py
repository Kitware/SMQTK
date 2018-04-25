import logging
from enum import IntEnum

try:
    import torch
except ImportError as ex:
    logging.getLogger(__name__).warning("Failed to import pytorch module: %s",
                                        str(ex))
    torch = None

class DIS_TYPE(IntEnum):
    L2 = 1
    hik = 2


def L2_dis(t1, t2, dim):
    # if not isinstance(t1, torch.Tensor):
    #     raise TypeError("{} has to be torch.Tensor!".format(t1))
    # if not isinstance(t2, torch.Tensor):
    #     raise TypeError("{} has to be torch.Tensor!".format(t2))

    res = (t1 - t2).norm(p=2, dim=dim)

    return res


def his_intersection_dis(t1, t2, dim):
    # if not isinstance(t1, torch.Tensor):
    #     raise TypeError("{} has to be torch.Tensor!".format(t1))
    # if not isinstance(t2, torch.Tensor):
    #     raise TypeError("{} has to be torch.Tensor!".format(t2))

    res = 1.0 - ((t1 + t2) - torch.abs(t1 - t2)).sum(dim=dim) * 0.5

    return res