import torchvision.models as models
import torch
from collections import OrderedDict
import os
from os import path as ospf
import sys
from contextlib import contextmanager

from transformers import ViTForImageClassification

this_dir = os.path.dirname(__file__)


def add_pythonpath(path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            sys.path.insert(0, path)
            original_pythonpath = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = f"{path}:{original_pythonpath}"
            try:
                result = func(*args, **kwargs)
            finally:
                os.environ["PYTHONPATH"] = original_pythonpath
                sys.path.pop(0)
            return result

        return wrapper

    return decorator


@contextmanager
def add_path_context(path):
    original_pythonpath = os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, path)
    os.environ["PYTHONPATH"] = f"{path}:{original_pythonpath}"
    try:
        yield
    finally:
        os.environ["PYTHONPATH"] = original_pythonpath
        sys.path.pop(0)


def remove_item_in_key(state_dict, item="module"):
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(item):
            k = k[(len(item) + 1):]
        new_dict[k] = v
    return new_dict


def load_resx50_org():
    from torchvision.models import resnext50_32x4d as x50
    model = x50(weights='DEFAULT')
    return model


class ViTHug_wrap(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(model_path)

    def forward(self, x, target=None):
        logits = self.model(x, target).logits
        return logits


def load_ft_vit_cifar100(model_path='pre_trained/vit_cifar100'):
    model = ViTHug_wrap(model_path)
    return model

#####################ncl为基础模型#########################

def load_ncl_cifar100(model_path='/root/autodl-tmp/label-shift-correction/train_lt_ncl/output/cifar100_im100/200_epoch/models/2024-12-04-11-22/best_ensemble_model.pth'):
    from .NCL_model.build_model import NCLModelWrapper
    pth_dict = torch.load(model_path)
    model = NCLModelWrapper()
    model.build_ncl_w_con_cifar100()

    state_dict = remove_item_in_key(pth_dict['state_dict'], 'module')

    model.model.load_state_dict(state_dict)
    return model


def load_ncl_cifar100_wo_con(model_path='C:/Users/hp/LSC/label-shift-correction/train_lt_ncl/output/cifar100_im100/baseline/models/2024-12-04-19-44/best_ensemble_model.pth', div_out=False, c_r_dir=None):
    from .NCL_model.build_model import NCLModelWrapper
    pth_dict = torch.load(model_path)
    model = NCLModelWrapper(div_out=div_out)
    model.build_ncl_wo_con_cifar100(c_r_dir=c_r_dir)
    # model.model = torch.nn.DataParallel(model.model).cuda()
    # state_dict = pth_dict['state_dict']
    state_dict = remove_item_in_key(pth_dict['state_dict'], 'module')
    model.model.load_state_dict(state_dict)
    return model


def load_ncl_imagenet(model_path='pre_trained/augmix/imagenet_ncl_augmix.pth', div_out=False):
    from .NCL_model.build_model import NCLModelWrapper
    pth_dict = torch.load(model_path)
    model = NCLModelWrapper(div_out=div_out)
    model.build_ncl_imagenet()
    # model.model = torch.nn.DataParallel(model.model).cuda()
    # state_dict = pth_dict['state_dict']
    state_dict = remove_item_in_key(pth_dict['state_dict'], 'module')
    model.model.load_state_dict(state_dict)
    return model


def load_ncl_imagenet_x50(model_path='pre_trained/imagenet/imagenet_ncl_x50.pth', div_out=False):
    from .NCL_model.build_model import NCLModelWrapper
    pth_dict = torch.load(model_path)
    model = NCLModelWrapper(div_out=div_out)
    model.build_ncl_imagenet_x50()
    # model.model = torch.nn.DataParallel(model.model).cuda()
    # state_dict = pth_dict['state_dict']
    state_dict = remove_item_in_key(pth_dict['state_dict'], 'module')
    model.model.load_state_dict(state_dict)
    return model

def load_ncl_imagenet_x50_wo_con(model_path='', div_out=False):
    from .NCL_model.build_model import NCLModelWrapper
    pth_dict = torch.load(model_path)
    model = NCLModelWrapper(div_out=div_out)
    model.build_ncl_imagenet_x50_wo_con()
    # model.model = torch.nn.DataParallel(model.model).cuda()
    # state_dict = pth_dict['state_dict']
    state_dict = remove_item_in_key(pth_dict['state_dict'], 'module')
    model.model.load_state_dict(state_dict)
    return model

def load_ncl_places_wo_con(model_path='pre_trained/places/NCL_places_wo_moco_hcm.pth', div_out=False):
    from .NCL_model.build_model import NCLModelWrapper
    pth_dict = torch.load(model_path)
    model = NCLModelWrapper(div_out=div_out)
    model.build_ncl_places_wo_con()
    # model.model = torch.nn.DataParallel(model.model).cuda()
    # state_dict = pth_dict['state_dict']
    state_dict = remove_item_in_key(pth_dict['state_dict'], 'module')
    model.model.load_state_dict(state_dict)
    return model

############


if __name__ == '__main__':
    pass
