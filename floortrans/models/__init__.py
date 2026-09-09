from floortrans.models.hg_furukawa_original import *

def get_model(name, n_classes=None, version=None, pretrained=True):
    if name == 'hg_furukawa_original':
        model = hg_furukawa_original(n_classes=n_classes)
        if pretrained:
            model.init_weights()
    else:
        raise ValueError('Model {} not available'.format(name))

    return model

