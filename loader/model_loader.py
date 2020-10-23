import json

import settings
import torch
import torchvision

from loader import binmatr2_multi_faces_resnet
from loader import binmatr2_multi_faces_resnet_NEW
from loader.resnet import ResidualNet

def loadmodel():
    if settings.MODEL_FILE is None:
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
        print('loaded pretrained model')
    else:
        if not (settings.MY_MODEL_CIFAR or settings.MY_MODEL_IMAGENETTE):
            checkpoint = torch.load(settings.MODEL_FILE)
            if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
                if not settings.MODEL == 'resnet50_mish':
                    model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
                else:
                    model = ResidualNet('ImageNet', 50, 1000)
                if settings.MODEL_PARALLEL:
                    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                        'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict)
            else:
                model = checkpoint
        else:
            # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_33_on_September_16/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl'
            # param_file = '/mnt/raid/data/chebykin/pycharm_project_AA/params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask.json'
            save_model_path = r'/mnt/raid/data/chebykin/saved_models/17_19_on_October_13/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl'
            param_file = '/mnt/raid/data/chebykin/pycharm_project_AA/params/binmatr2_imagenette_sgd1bias_fc_batch128_weightdecay3e-4_singletask.json'
            with open(param_file) as json_params:
                params = json.load(json_params)
            state = torch.load(save_model_path)
            model_rep_state = state['model_rep']
            if False:
                model = binmatr2_multi_faces_resnet.BinMatrResNet(binmatr2_multi_faces_resnet.BasicBlock,
                                                                         [2, 2, 2, 2],
                                                                         params['chunks'], 1,
                                                                         params['if_fully_connected'], True,
                                                                         10)
            else:
                model = binmatr2_multi_faces_resnet_NEW.BinMatrResNet(binmatr2_multi_faces_resnet_NEW.BasicBlock,
                                                                  [2, 2, 2, 2],
                                                                  params['chunks'], 1,
                                                                  params['if_fully_connected'], 'imagenette',
                                                                  10, if_enable_bias=params['if_enable_bias'])
            model.load_state_dict(model_rep_state)

    for name in settings.FEATURE_NAMES:
        if False:
            model._modules.get(name).register_forward_hook(settings.HOOK_FN)
        else:
            dict(model.named_modules()).get(name).register_forward_hook(settings.HOOK_FN)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model
