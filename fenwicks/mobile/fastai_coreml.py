import torch
import torch.onnx
import torch.nn as nn
import onnx
from onnx_coreml import convert
from torch.autograd import Variable
from typing import List, Dict


class ImageScale(nn.Module):
    def __init__(self):
        super().__init__()
        imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.mean, self.std = torch.tensor(imagenet_stats).cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = x / self.std[..., None, None]
        return xx.unsqueeze(0)


def learner_pytorch(learner):
    model = [ImageScale()] + (list(learner.model.children())[:])
    return nn.Sequential(*model)


def export_onnx_and_check(model, model_name: str = 'model.onnx', sz: int = 224, input_names: List[str] = None,
                          output_names: List[str] = None):
    input_names = input_names or ['input']
    output_names = output_names or ['output']

    x = torch.randn(3, sz, sz)
    dummy_input = Variable(x).cuda()

    torch.onnx.export(model, dummy_input, model_name, input_names=input_names, output_names=output_names, verbose=True)

    onnx_model = onnx.load(model_name)
    onnx.checker.check_model(onnx_model)

    return onnx_model


def onnx_coreml(onnx_model, model_name: str = 'model.mlmodel', input_names: List[str] = None,
                label_fn: str = 'labels.txt', author: str = 'david', license_type: str = 'MIT', short_desc: str = '',
                input_desc: Dict[str, str] = None, output_desc: Dict[str, str] = None):
    input_names = input_names or ['input']
    scale = 1.0 / 255.0

    args = dict(is_bgr=False, red_bias=-0.485, green_bias=-0.456, blue_bias=-0.406, image_scale=scale)
    mlmodel = convert(onnx_model, image_input_names=input_names, mode='classifier', class_labels=label_fn,
                      preprocessing_args=args)

    mlmodel.author = author
    mlmodel.license = license_type
    mlmodel.short_description = short_desc

    for k, v in input_desc:
        mlmodel.input_description[k] = v

    for k, v in output_desc:
        mlmodel.output_description[k] = v

    mlmodel.save(model_name)

    return mlmodel
