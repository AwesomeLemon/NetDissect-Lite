from feature_operation import hook_feature_output, hook_feature_input
######### global settings  #########
GPU = True                                  # running on GPU is highly suggested
TEST_MODE = False                           # turning on the testmode means the code will run on a small dataset.
CLEAN = False                               # set to "True" if you want to clean the temporary large files after generating result
MODEL = 'efficientnet-b3'#'resnet18'#'mobilenet_v2'#'resnet50_mish'#'shufflenet_v2_x1_0'#'wide_resnet50_2'#'vgg19'                          # model arch: resnet18, alexnet, resnet50, densenet161
DATASET = 'imagenet'# 'places365'#                      # model trained on: places365 or imagenet
QUANTILE = 0.005#0.05#                          # the threshold used for activation
SEG_THRESHOLD = 0.04                        # the threshold used for visualization
SCORE_THRESHOLD = 0.04                      # the threshold used for IoU score (in HTML file)
TOPN = 9                                   # to show top N image with highest activation for each unit
PARALLEL = 2                               # how many process is used for tallying (Experiments show that 1 is the fastest)
CATAGORIES = ["object", "part","scene", "color","texture"] #"color","texture" concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
OUTPUT_FOLDER = "result/pytorch_"+MODEL+"_"+DATASET # result will be stored in this folder
OUTPUT_FOLDER += '_69'
LOOK_AT_MAX = False
MY_MODEL_CIFAR = False
MY_MODEL_IMAGENETTE = False
HOOK_FN = hook_feature_output#hook_feature_input#

########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# FEATURE_NAMES: the array of layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own

if MODEL != 'alexnet':
    DATA_DIRECTORY = 'dataset/broden1_224'
    IMG_SIZE = 224
else:
    DATA_DIRECTORY = 'dataset/broden1_227'
    IMG_SIZE = 227
    FEATURE_NAMES = ['features']
    MODEL_FILE = None
    MODEL_PARALLEL = False

if DATASET == 'places365':
    NUM_CLASSES = 365
elif DATASET == 'imagenet':
    NUM_CLASSES = 1000
if MODEL == 'resnet18':
    FEATURE_NAMES = ['layer3.1.relu2']#['layer1.0']#['layer1.0.relu2']#['layer4.1.relu2']#
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/resnet18_places365.pth.tar'
        MODEL_PARALLEL = True
    elif DATASET == 'imagenet':
        MODEL_FILE = None#'i wanna use my model'
        MODEL_PARALLEL = False
elif MODEL == 'densenet161':
    FEATURE_NAMES = ['features']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/whole_densenet161_places365_python36.pth.tar'
        MODEL_PARALLEL = False
elif MODEL == 'resnet50':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/whole_resnet50_places365_python36.pth.tar'
        MODEL_PARALLEL = False
elif 'vgg' in MODEL:
    FEATURE_NAMES = ['features']
    MODEL_FILE = None
    MODEL_PARALLEL = False
elif 'wide_resnet50_2' in MODEL:
    FEATURE_NAMES = ['layer3']
    MODEL_FILE = None
    MODEL_PARALLEL = False
elif 'resnet152' in MODEL:
    FEATURE_NAMES = ['layer4.1.conv1']
    MODEL_FILE = None
    MODEL_PARALLEL = False
elif 'shufflenet_v2_x1_0' in MODEL:
    FEATURE_NAMES = ['stage4']
    MODEL_FILE = None
    MODEL_PARALLEL = False
elif 'resnet50_mish' in MODEL:
    FEATURE_NAMES = ['layer4']
    MODEL_FILE = 'zoo/RS50_ACT_model_best.pth.tar'
    MODEL_PARALLEL = True
elif 'mobilenet_v2' in MODEL:
    FEATURE_NAMES = ['features.17']#['features']
    MODEL_FILE = None
    MODEL_PARALLEL = False
elif 'efficientnet-b3' in MODEL:
    FEATURE_NAMES = ['_blocks.25']#['_blocks.25.pre_project_conv']#
    MODEL_FILE = None
    MODEL_PARALLEL = False

# if TEST_MODE:
#     WORKERS = 1
#     BATCH_SIZE = 4
#     TALLY_BATCH_SIZE = 2
#     TALLY_AHEAD = 1
#     INDEX_FILE = 'index_sm.csv'
#     OUTPUT_FOLDER += "_test"
# else:
#     WORKERS = 12
#     BATCH_SIZE = 128
#     TALLY_BATCH_SIZE = 16
#     TALLY_AHEAD = 4
#     INDEX_FILE = 'index.csv'

WORKERS = 12
BATCH_SIZE = 24#32#128
TALLY_BATCH_SIZE = 64
TALLY_AHEAD = 4
if TEST_MODE:
    # INDEX_FILE = 'index_sm.csv'
    INDEX_FILE = 'index_sm_orig.csv'
    OUTPUT_FOLDER += "_test"
else:
    INDEX_FILE = 'index.csv'

