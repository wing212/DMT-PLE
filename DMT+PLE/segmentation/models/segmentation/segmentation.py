from .._utils import IntermediateLayerGetter
from ..utils import load_state_dict_from_url
from .. import resnet
from .deeplab import DeepLabV3Head, DeepLabV2Head, DeepLab, ReconHead
from .fcn import FCN, FCNHead


__all__ = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv2_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101',
           'deeplab_v2']


model_urls = {
    'fcn_resnet50_coco': None,
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': None,
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
    'deeplabv2_resnet101_coco': None,
}


def _segm_resnet(name, backbone_name, num_classes, aux, recon_loss, pretrained_backbone=True):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    if recon_loss:
        return_layers['layer2'] = 'recon'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    recon_classifier = None
    if recon_loss:
        recon_classifier = ReconHead(in_channels=512)

    model_map = {
        'deeplabv3': (DeepLabV3Head, DeepLab),
        'deeplabv2': (DeepLabV2Head, DeepLab),
        'fcn': (FCNHead, FCN),
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier, recon_classifier)
    return model


def _load_model(arch_type, backbone, pretrained, progress, num_classes, aux_loss, recon_loss, **kwargs):
    if pretrained:
        aux_loss = True
    model = _segm_resnet(arch_type, backbone, num_classes, aux_loss, recon_loss, **kwargs)
    if pretrained:
        arch = arch_type + '_' + backbone + '_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model


def fcn_resnet50(pretrained=False, progress=True,
                 num_classes=21, aux_loss=None, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('fcn', 'resnet50', pretrained, progress, num_classes, aux_loss, **kwargs)


def fcn_resnet101(pretrained=False, progress=True,
                  num_classes=21, aux_loss=None, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('fcn', 'resnet101', pretrained, progress, num_classes, aux_loss, **kwargs)


def deeplabv2_resnet101(pretrained=False, progress=True,
                        num_classes=21, aux_loss=None, recon_loss=False, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv2', 'resnet101', pretrained, progress, num_classes, aux_loss,
                       recon_loss, **kwargs)


def deeplabv3_resnet50(pretrained=False, progress=True,
                       num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet50', pretrained, progress, num_classes, aux_loss, **kwargs)


def deeplabv3_resnet101(pretrained=False, progress=True,
                        num_classes=21, aux_loss=None, recon_loss=False, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet101', pretrained, progress, num_classes, aux_loss,
                       recon_loss, **kwargs)


def deeplab_v2(num_classes):
    # Define deeplabV2 with ResNet101(With only ImageNet pretraining)
    return deeplabv2_resnet101(pretrained=False, num_classes=num_classes, recon_loss=False)
