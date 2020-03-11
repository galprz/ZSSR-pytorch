import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F

class VGGActivationExtractor(nn.Module):
    def __init__(self,
                 layer_name_mapping,
                 device):
        super(VGGActivationExtractor, self).__init__()
        self.vgg_model = vgg16(pretrained=True)
        self.vgg_model.eval()
        self.vgg_model.to(device)
        self.vgg_layers = self.vgg_model.features
        self.layer_name_mapping = layer_name_mapping
        self.number_of_layers = len(self.layer_name_mapping)
        self.eval()

    def forward(self, x):
        output = []
        layer_count = 0
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x.clone())
                layer_count += 1
            if layer_count == self.number_of_layers:
                break
        return output

class ContentLoss(nn.Module):
    def __init__(self,
                 layer_name_mapping=None,
                 layer_weights=None,
                 device="cuda"):
        super(ContentLoss, self).__init__()
        self.layer_name_mapping = layer_name_mapping
        if layer_name_mapping is None:
            self.layer_name_mapping = {
                '3': "relu1_2",
                '8': "relu2_2",
                '15': "relu3_3",
            }
        self.layer_weights = layer_weights
        if layer_weights is None:
            self.layer_weights = [1. / len(self.layer_name_mapping)] * len(self.layer_name_mapping)
        weights_sum = sum(self.layer_weights)
        if weights_sum > 1:
            # normalize weights
            self.layer_weights = [w / weights_sum for w in self.layer_weights]

        self.activation_extractor = VGGActivationExtractor(self.layer_name_mapping, device)

    def forward(self, y, y_pred):
        y_activations = self.activation_extractor(y)
        y_pred_activations = self.activation_extractor(y_pred)
        losses = [F.l1_loss(y_layer_activations, y_pred_layer_activations) * layer_weight
                  for y_layer_activations, y_pred_layer_activations, layer_weight
                  in zip(y_activations, y_pred_activations, self.layer_weights)]
        return sum(losses)