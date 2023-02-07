import torch
import argparse

from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
from mqbench.convert_deploy import convert_deploy             # remove quant nodes for deploy

def parse_args():
    parser = argparse.ArgumentParser(description='Test Multitask network')

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='runs/')
    parser.add_argument('--weights', nargs='+', type=str, default='/data2/zwt/wd/YOLOP/runs/BddDataset/detect_and_segbranch_whole/epoch-169.pth', help='model.pth path(s)')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()

    return args

args = parse_args()
update_config(cfg, args)

print("begin to bulid up model...")
# DP mode
device = select_device(logger, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU* len(cfg.GPUS)) if not cfg.DEBUG \
    else select_device(logger, 'cpu')
# device = select_device(logger, 'cpu')

model = get_net(cfg)
print("build model 1/2")

# define loss function (criterion) and optimizer
criterion = get_loss(cfg, device=device)
optimizer = get_optimizer(cfg, model)

# load checkpoint model
# det_idx_range = [str(i) for i in range(0,25)]
model_dict = model.state_dict()
# NOTE: Since 'weights' is a list, we need to read the element.
checkpoint_file = args.weights[0]
logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
# NOTE: Adaption for CPU execution.
checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
checkpoint_dict = checkpoint['state_dict']
model_dict.update(checkpoint_dict)
model.load_state_dict(model_dict)
logger.info("=> loaded checkpoint '{}' ".format(checkpoint_file))

model = model.to(device)

model.gr = 1.0
model.nc = 1

print('bulid model 2/2')
