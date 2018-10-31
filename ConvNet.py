import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class ConvNet(nn.Module):
  def __init__(self, num_classes=2):
    super(ConvNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(9, 32, kernel_size=11, stride=4, padding=2),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.LPPool2d(1, kernel_size=2, stride=2),
      nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.fc = nn.Sequential(
      nn.Linear(6 * 6 * 64, 256),
      nn.Sigmoid(),
      nn.Linear(256, num_classes))

  def forward(self, x):
    out = self.features(x)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    return out


class ConvNet_v1(nn.Module):
  def __init__(self, num_classes=2):
    super(ConvNet_v1, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(9, 32, kernel_size=11, stride=4),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.LPPool2d(1, kernel_size=2, stride=2),
      nn.Conv2d(32, 64, kernel_size=5, stride=2),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(64, 64, kernel_size=3),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.fc = nn.Sequential(
      nn.Linear(4 * 4 * 64, 256),
      nn.Sigmoid(),
      nn.Linear(256, num_classes))

  def forward(self, x):
    out = self.features(x)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    return out


class ConvNet_v2(nn.Module):
  def __init__(self, num_classes=2):
    super(ConvNet_v2, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(9, 16, kernel_size=11),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True),
      nn.LPPool2d(1, kernel_size=2, stride=2),
      nn.Conv2d(16, 32, kernel_size=5),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(32, 64, kernel_size=3),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(64, 48, kernel_size=3),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(48, 48, kernel_size=3),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.fc = nn.Sequential(
      nn.Linear(10 * 10 * 48, 1024),
      nn.Sigmoid(),
      nn.Linear(1024, 256),
      nn.Sigmoid(),
      nn.Linear(256, num_classes))

  def forward(self, x):
    out = self.features(x)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    return out


class ConvNet_v3(nn.Module):
  def __init__(self, num_classes=2):
    super(ConvNet_v3, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(9, 32, kernel_size=5),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.LPPool2d(1, kernel_size=2, stride=2),
      nn.Conv2d(32, 64, kernel_size=5),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(64, 128, kernel_size=3),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(128, 128, kernel_size=3),
      nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveMaxPool2d(7)
    self.fc = nn.Sequential(
      nn.Linear(7 * 7 * 128, 256),
      nn.Sigmoid(),
      nn.Linear(256, num_classes))

  def forward(self, x):
    out = self.features(x)
    out = self.global_pooling(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    return out


class ConvNet_v4(nn.Module):
  def __init__(self, num_classes=2):
    super(ConvNet_v4, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(9, 16, kernel_size=5),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True),
      nn.LPPool2d(1, kernel_size=2, stride=2))
    self.layer2 = nn.Sequential(
      nn.Conv2d(16, 24, kernel_size=5),
      nn.BatchNorm2d(24),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer3 = nn.Sequential(
      nn.Conv2d(72, 128, kernel_size=3),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer4 = nn.Sequential(
      nn.Conv2d(128, 128, kernel_size=3),
      nn.ReLU(inplace=True),
      nn.AdaptiveMaxPool2d(7))
    self.fc = nn.Sequential(
      nn.Linear(7 * 7 * 128, 256),
      nn.Sigmoid(),
      nn.Linear(256, num_classes))

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    broadcast_h = F.adaptive_max_pool1d(out.squeeze(), 1)
    broadcast_v = F.adaptive_max_pool1d(out.squeeze().permute(0, 2, 1), 1)
    out = torch.unsqueeze(torch.cat(
      (out.squeeze(), broadcast_h.repeat(1, 1, out.size(3)), broadcast_v.permute(0, 2, 1).repeat(1, out.size(3), 1)),
      0), 0)
    out = self.layer3(out)
    out = self.layer4(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    return out


class ConvNet_v5(nn.Module):
  def __init__(self, num_classes=2):
    super(ConvNet_v5, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(9, 16, kernel_size=7, stride=3, padding=2),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer2 = nn.Sequential(
      nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(48),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer3 = nn.Sequential(
      nn.Conv2d(144, 144, kernel_size=3, padding=1),
      nn.ReLU(inplace=True))
    self.layer4 = nn.Sequential(
      nn.Conv2d(432, 432, kernel_size=3, padding=1),
      nn.ReLU(inplace=True))
    self.layer5 = nn.AdaptiveMaxPool2d(2)
    self.fc = nn.Sequential(
      nn.Linear(2 * 2 * 432, 256),
      nn.Sigmoid(),
      nn.Linear(256, num_classes))

  def forward(self, x):
    out = self.layer1(x)
    broadcast_h = F.adaptive_max_pool1d(out.squeeze(), 1)
    broadcast_v = F.adaptive_max_pool1d(out.squeeze().permute(0, 2, 1), 1)
    out = torch.unsqueeze(torch.cat(
      (out.squeeze(), broadcast_h.repeat(1, 1, out.size(3)), broadcast_v.permute(0, 2, 1).repeat(1, out.size(3), 1)),
      0), 0)
    out = self.layer2(out)
    broadcast_h = F.adaptive_max_pool1d(out.squeeze(), 1)
    broadcast_v = F.adaptive_max_pool1d(out.squeeze().permute(0, 2, 1), 1)
    out = torch.unsqueeze(torch.cat(
      (out.squeeze(), broadcast_h.repeat(1, 1, out.size(3)), broadcast_v.permute(0, 2, 1).repeat(1, out.size(3), 1)),
      0), 0)
    out = self.layer3(out)
    broadcast_h = F.adaptive_max_pool1d(out.squeeze(), 1)
    broadcast_v = F.adaptive_max_pool1d(out.squeeze().permute(0, 2, 1), 1)
    out = torch.unsqueeze(torch.cat(
      (out.squeeze(), broadcast_h.repeat(1, 1, out.size(3)), broadcast_v.permute(0, 2, 1).repeat(1, out.size(3), 1)),
      0), 0)
    out = self.layer4(out)
    out = self.layer5(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    return out


class ConvNet_v6(nn.Module):
  def __init__(self, num_classes=2):
    super(ConvNet_v6, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(9, 32, kernel_size=11, stride=4, padding=10),
      nn.ReLU(inplace=True),
      nn.LPPool2d(1, kernel_size=2, stride=2))
    self.layer2 = nn.Sequential(
      nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=4),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer3 = nn.Sequential(
      nn.Conv2d(64, 64, kernel_size=3, padding=1),
      nn.ReLU(inplace=True))
    self.layer4 = nn.AdaptiveAvgPool2d(4)
    self.fc = nn.Sequential(
      nn.Linear(4 * 4 * 64, 256),
      nn.Sigmoid(),
      nn.Linear(256, num_classes))

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    return out


class ConvNet_v7(nn.Module):
  def __init__(self, num_classes=2):
    super(ConvNet_v7, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(9, 32, kernel_size=11, stride=4, padding=10),
      nn.ReLU(inplace=True),
      nn.LPPool2d(1, kernel_size=2, stride=2))
    self.layer2 = nn.Sequential(
      nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=4),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer3 = nn.Sequential(
      nn.Conv2d(64, 64, kernel_size=3, padding=1),
      nn.ReLU(inplace=True))
    self.layer4 = nn.AdaptiveAvgPool2d(1)
    self.layer5 = nn.AdaptiveAvgPool2d(2)
    self.layer6 = nn.AdaptiveAvgPool2d(4)
    self.fc = nn.Sequential(
      nn.Linear((1 * 1 + 2 * 2 + 4 * 4) * 64, 256),
      nn.Sigmoid(),
      nn.Linear(256, num_classes))

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out1 = self.layer4(out)
    out2 = self.layer5(out)
    out3 = self.layer6(out)
    out = torch.cat((out1.reshape(out1.size(0), -1), out2.reshape(out2.size(0), -1), out3.reshape(out3.size(0), -1)), 1)
    out = self.fc(out)
    return out


class AlexNet(nn.Module):
  """
  reference: https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
  """

  def __init__(self, num_classes=2):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(9, 64, kernel_size=11, stride=4, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=5, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.AdaptiveAvgPool2d(6),
    )
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 6 * 6, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256 * 6 * 6)
    x = self.classifier(x)
    return x


model_urls = {
  'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


def alexnet(pretrained=False, **kwargs):
  """AlexNet model architecture from the
  `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """

  model = AlexNet(**kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
  return model


def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  """
  reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
  """

  def __init__(self, block, layers, num_classes=2):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    self.avgpool = nn.AvgPool2d(7, stride=1)
    self.fc = nn.Linear(7 * 7 * 512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet18(pretrained=False, **kwargs):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False, **kwargs):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False, **kwargs):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False, **kwargs):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model
