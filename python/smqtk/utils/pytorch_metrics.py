import logging
from enum import Enum

try:
    import torch
except ImportError as ex:
    logging.getLogger(__name__).warning("Failed to import pytorch module: %s",
                                        str(ex))
    torch = None

class DIS_TYPE(Enum):
    L2 = 1
    hik = 2


def L2_dis(t1, t2, dim):
    #print(t2.dtype())
    #print("type:t2",dtype(t2))
    #if type(t1)!=torch.Tensor:
    #   t1=t1.double  
        #raise TypeError("{} has to be torch.Tensor!".format(t1))
    #if not isinstance(t2, torch.Tensor.float()):
    #      t2=t2.float()
    #     raise TypeError("{} has to be torch.Tensor!".format(t2))
    res = (t1.float() - t2.float()).norm(p=2, dim=dim)

    return res


def his_intersection_dis(t1, t2, dim):
    # if not isinstance(t1, torch.Tensor):
    #     raise TypeError("{} has to be torch.Tensor!".format(t1))
    # if not isinstance(t2, torch.Tensor):
    #     raise TypeError("{} has to be torch.Tensor!".format(t2))

    res = 1.0 - ((t1 + t2) - torch.abs(t1 - t2)).sum(dim=dim) * 0.5

    return res
