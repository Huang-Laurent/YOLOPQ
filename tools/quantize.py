import argparse
import os, sys
import math
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import time
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from lib.utils import DataLoaderX, torch_distributed_zero_first
from tensorboardX import SummaryWriter

import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import train
from lib.core.function import validate
from lib.core.general import fitness
from lib.models import get_net
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device
from lib.utils import run_anchor

from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
from mqbench.convert_deploy import convert_deploy             # remove quant nodes for deploy

from lib.models.common2 import Detect
from torch.nn import Sigmoid

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

def main():
    args = parse_args()
    update_config(cfg, args)


    # logger, final_output_dir, tb_log_dir = create_logger(
    #     cfg, cfg.LOG_DIR, 'train', rank=rank)
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'test')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    print("begin to bulid up model...")
    # DP mode
    # device = select_device(logger, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU* len(cfg.GPUS)) if not cfg.DEBUG \
    #     else select_device(logger, 'cpu')
    device = select_device(logger, 'cpu')

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
    # checkpoint = torch.load(checkpoint_file, map_location=torch.device('cuda'))
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    checkpoint_dict = checkpoint['state_dict']
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    logger.info("=> loaded checkpoint '{}' ".format(checkpoint_file))

    model = model.to(device)
    model.gr = 1.0
    model.nc = 1

    print('bulid model 2/2')

    extra_qconfig_dict = {
        'w_observer': 'MinMaxObserver',
        'a_observer': 'EMAMinMaxObserver',
        'w_fakequantize': 'FixedFakeQuantize',
        'a_fakequantize': 'FixedFakeQuantize',
    }
    
    # leaf_module = (Detect, Sigmoid, )
    # leaf_module = (Sigmoid, )
    # prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict, 'leaf_module':leaf_module}
    prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
    print('prepare quantize model 1')
    backend = BackendType.Tensorrt
    model.eval()
    print('prepare quantize model 2')
    model = prepare_by_platform(model, backend, prepare_custom_config_dict)  
    print('prepare quantize model 3')
    enable_calibration(model) 

    print("begin to load data")
    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    # valid_loader = DataLoaderX(
    #     valid_dataset,
    #     batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
    #     shuffle=False,
    #     num_workers=cfg.WORKERS,
    #     pin_memory=cfg.PIN_MEMORY,
    #     collate_fn=dataset.AutoDriveDataset.collate_fn
    # )
    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=False,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    print('load data finished')

    ### Evaluate the original model.
    print("\n----- EVALUATION OF A NON COMPRESSED MODEL -----")

    epoch = 0 #special for test
    da_segment_results,ll_segment_results,detect_results, total_loss,maps, times = validate(
        epoch,cfg, valid_loader, valid_dataset, model, criterion,
        final_output_dir, tb_log_dir, writer_dict,
        logger, device
    )
    fi = fitness(np.array(detect_results).reshape(1, -1))
    msg =   'Test:    Loss({loss:.3f})\n' \
            'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                      'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                      'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
                      'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                          loss=total_loss, da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                          ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2],
                          p=detect_results[0],r=detect_results[1],map50=detect_results[2],map=detect_results[3],
                          t_inf=times[0], t_nms=times[1])
    logger.info(msg)

    print('quantize model')
    enable_quantization(model)
    print('quantize model done')

    print("\n----- EVALUATION OF POST-TRAINING QUANTIZED MODEL -----")

    da_segment_results_ptq,ll_segment_results_ptq,detect_results_ptq,total_loss_ptq,maps_ptq,times_ptq = validate(
        epoch, cfg, valid_loader, valid_dataset, model, criterion,
        final_output_dir, tb_log_dir, writer_dict,
        logger, device
    )
    fi_ptq  = fitness(np.array(detect_results_ptq).reshape(1, -1))
    msg_ptq = 'Test:    Loss({loss:.3f})\n' \
              'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                      'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                      'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
                      'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                          loss=total_loss_ptq, da_seg_acc=da_segment_results_ptq[0],da_seg_iou=da_segment_results_ptq[1],da_seg_miou=da_segment_results_ptq[2],
                          ll_seg_acc=ll_segment_results_ptq[0],ll_seg_iou=ll_segment_results_ptq[1],ll_seg_miou=ll_segment_results_ptq[2],
                          p=detect_results_ptq[0],r=detect_results_ptq[1],map50=detect_results_ptq[2],map=detect_results_ptq[3],
                          t_inf=times_ptq[0], t_nms=times_ptq[1])
    logger.info(msg_ptq)
    print("test finish")

if __name__ == '__main__':
    main()
