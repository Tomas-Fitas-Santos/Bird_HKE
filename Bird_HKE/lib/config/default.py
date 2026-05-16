from yacs.config import CfgNode as CN
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.CKPT_DIR = ''
_C.CKPT_FILE = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.RESUME_FROM_CKPT = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 4
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False
_C.LOSS.USE_FOCAL = False
_C.LOSS.FOCAL_ALPHA = 2.0
_C.LOSS.FOCAL_BETA = 4.0
_C.LOSS.FOCAL_WEIGHT = 1.0
_C.LOSS.MSE_WEIGHT = 1.0

# DATASET related params
_C.DATASET = CN()
_C.DATASET.NAME_ = 'birdgaze.'  #'coco.' 'ak.'
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = ''
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.NUM_JOINTS_HALF_BODY = 2
_C.DATASET.COLOR_RGB = False

def _build_phase_cfg():
    phase = CN()

    phase.LR_FACTOR = 0.1
    phase.LR_STEP = [90, 110]
    phase.LR = 0.001

    phase.OPTIMIZER = 'adam'
    phase.MOMENTUM = 0.9
    phase.WD = 0.0001
    phase.NESTEROV = False
    phase.GAMMA1 = 0.99
    phase.GAMMA2 = 0.0
    phase.LR_SCHEDULE = 'multistep'
    phase.WARMUP_EPOCHS = 0
    phase.MIN_LR = 0.0
    phase.CLIP_GRAD_NORM = 0.0
    phase.GRAD_ACCUM_STEPS = 1

    phase.BEGIN_EPOCH = 0
    phase.END_EPOCH = 140

    phase.RESUME = False
    phase.CHECKPOINT = ''
    phase.RESUME_FROM_CKPT = False
    phase.CKPT_DIR = ''
    phase.LOG_DIR = ''

    phase.BATCH_SIZE_PER_GPU = 32
    phase.SHUFFLE = True
    return phase


# train / finetune
_C.TRAIN = _build_phase_cfg()

_C.FINETUNE = _build_phase_cfg()
_C.FINETUNE.DATA_DIR = ''
_C.FINETUNE.SOURCE_DIR = ''
_C.FINETUNE.SOURCE_MODEL_FILE = ''
_C.FINETUNE.CKPT_FILE = 'finetune_checkpoint.pth'
_C.FINETUNE.BEST_MODEL_FILE = 'finetune_model_best.pth'
_C.FINETUNE.FINAL_MODEL_FILE = 'finetune_final_model.pth'

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False

_C.TEST.USE_GT_BBOX = False

# nms
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.POSE_MODEL_FILE = ''
_C.TEST.DETECT_MODEL_FILE = ''
_C.TEST.OUTPUT_DIR = ''
_C.TEST.SCORE_ACTIVATION = 'sigmoid'  # 'none' or 'sigmoid'
_C.TEST.SCORE_TEMPERATURE = 1.0
_C.TEST.SCORE_CLIP = True
_C.TEST.SCORE_MODE = 'entropy'  # 'max' or 'entropy'
_C.TEST.SCORE_ENTROPY_BETA = 1.0
_C.TEST.SCORE_EPS = 1e-6

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    # If TRAIN.LOG_DIR is provided in the YAML, use it as the canonical
    # `LOG_DIR` so older code paths that read cfg.LOG_DIR continue to work.
    if hasattr(cfg, 'TRAIN') and hasattr(cfg.TRAIN, 'LOG_DIR') and cfg.TRAIN.LOG_DIR:
        cfg.LOG_DIR = cfg.TRAIN.LOG_DIR

    # If TEST.OUTPUT_DIR is provided in the YAML and no top-level OUTPUT_DIR
    # was set (for example via --modelDir/args), use it as the canonical
    # `OUTPUT_DIR` for test/inference results.
    if hasattr(cfg, 'TEST') and hasattr(cfg.TEST, 'OUTPUT_DIR') and cfg.TEST.OUTPUT_DIR and not cfg.OUTPUT_DIR:
        cfg.OUTPUT_DIR = cfg.TEST.OUTPUT_DIR

    # DATA_DIR removed: use DATASET.ROOT directly as the dataset root path.
    # Normalize DATASET.ROOT to an absolute path.
    cfg.DATASET.ROOT = os.path.expanduser(cfg.DATASET.ROOT)
    if cfg.DATASET.ROOT and not os.path.isabs(cfg.DATASET.ROOT):
        cfg.DATASET.ROOT = os.path.abspath(cfg.DATASET.ROOT)

    # Derive checkpoint directory: prefer explicit CKPT_DIR in config; if
    # relative, join with LOG_DIR. If not set, default to
    # <cwd>/checkpoints (do NOT use OUTPUT_DIR for training artifacts).
    try:
        # prefer TRAIN.CKPT_DIR when provided in experiments YAMLs
        if hasattr(cfg, 'TRAIN') and hasattr(cfg.TRAIN, 'CKPT_DIR') and cfg.TRAIN.CKPT_DIR:
            ckpt = cfg.TRAIN.CKPT_DIR
        else:
            ckpt = cfg.CKPT_DIR if hasattr(cfg, 'CKPT_DIR') else ''
    except Exception:
        ckpt = ''

    if ckpt:
        # If user provided a relative path, make it relative to LOG_DIR if set,
        # otherwise make it relative to the current working directory.
        if not os.path.isabs(ckpt):
            if cfg.LOG_DIR:
                cfg.CKPT_DIR = os.path.join(cfg.LOG_DIR, ckpt)
            else:
                cfg.CKPT_DIR = os.path.abspath(os.path.join(os.getcwd(), ckpt))
        else:
            cfg.CKPT_DIR = ckpt
    else:
        # default fallback: prefer LOG_DIR for training-related checkpoints
        if cfg.LOG_DIR:
            cfg.CKPT_DIR = os.path.join(cfg.LOG_DIR, 'checkpoints')
        else:
            cfg.CKPT_DIR = os.path.join(os.getcwd(), 'checkpoints')

    # If a user set TRAIN.RESUME_FROM_CKPT in their YAML, propagate it to
    # the top-level `RESUME_FROM_CKPT` so legacy call sites can read it.
    if hasattr(cfg, 'TRAIN') and hasattr(cfg.TRAIN, 'RESUME_FROM_CKPT'):
        cfg.RESUME_FROM_CKPT = cfg.TRAIN.RESUME_FROM_CKPT

    cfg.freeze()


# Detection / interpolation controls
_C.DETECTION = CN()
# Maximum number of consecutive frames to interpolate between two real
# detections. Gaps larger than this will not be linearly interpolated and
# will remain empty (no synthetic boxes). Set to 0 to disable interpolation.
_C.DETECTION.MAX_INTERPOLATION_GAP = 2
# Maximum number of frames at the start/end that may be filled by copying
# the first/last real detection. Set to 0 to disable endpoint filling.
_C.DETECTION.MAX_ENDPOINT_FILL = 2



