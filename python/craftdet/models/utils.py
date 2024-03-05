import torch
import torch.nn as nn


def init_weights(modules):
  """
  Initial weights of modules.
  """
  for m in modules:
    if isinstance(m, nn.Conv2d):
      torch.nn.init.xavier_uniform_(m.weight.data)
    if m.bias is not None:
      m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
      m.weight.data.fill_(1)
      m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
      m.weight.data.normal_(0, 0.01)
      m.bias.data.zero_()