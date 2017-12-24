# !/usr/bin/env python

# --------------------------------------------------------
# Written by Bharat Singh
# Modified version of py-R-FCN
# --------------------------------------------------------

#"""Train a Fast R-CNN network on a region of interest database."""

# import _init_paths
import sys

sys.path.append('/home/priv-lab1/workspace/py-RFCN-priv/caffe-priv/python')
sys.path.append('/home/priv-lab1/workspace/py-RFCN-priv/lib')

from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from utils.timer import Timer
import caffe
import argparse
import pprint
import numpy as np
import sys
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format
from multiprocessing import Process


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument("--gpu_id", type=str,
                        default='0,1,2,3,4,5,6,7',
                        help="List of device ids.")
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default='./solver.prototxt', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=160000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default='/home/priv-lab1/Program/caffe-model/cls/senet/se-resnet50/se-resnet50-hik-merge.caffemodel',
                        type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./faster_rcnn_end2end.yml', type=str)
    parser.add_argument('--out', dest='out_dir',
                        help='Output will be saved to',
                        default='./', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, output_dir, gpu_id,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self.gpu_id = gpu_id
        self.num_classes = len(open(cfg.TRAIN.LABELMAP_PATH, 'r').read().splitlines())

        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
            cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            num_reg_classes = 2 if cfg.TRAIN.AGNOSTIC else self.num_classes
            means = np.tile(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (num_reg_classes, 1))
            stds = np.tile(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (num_reg_classes, 1))
            self.bbox_means = means.ravel()
            self.bbox_stds = stds.ravel()
            print 'done'

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].get_gpu_id(gpu_id)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        scale_bbox_params_faster_rcnn = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             net.params.has_key('bbox_pred'))

        scale_bbox_params_rfcn = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             net.params.has_key('rfcn_bbox'))

        scale_bbox_params_rpn = (cfg.TRAIN.RPN_NORMALIZE_TARGETS and
                                 net.params.has_key('rpn_bbox_pred'))

        if scale_bbox_params_faster_rcnn:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        if scale_bbox_params_rpn:
            rpn_orig_0 = net.params['rpn_bbox_pred'][0].data.copy()
            rpn_orig_1 = net.params['rpn_bbox_pred'][1].data.copy()
            num_anchor = rpn_orig_0.shape[0] / 4
            # scale and shift with bbox reg unnormalization; then save snapshot
            self.rpn_means = np.tile(np.asarray(cfg.TRAIN.RPN_NORMALIZE_MEANS),
                                      num_anchor)
            self.rpn_stds = np.tile(np.asarray(cfg.TRAIN.RPN_NORMALIZE_STDS),
                                     num_anchor)
            net.params['rpn_bbox_pred'][0].data[...] = \
                (net.params['rpn_bbox_pred'][0].data *
                 self.rpn_stds[:, np.newaxis, np.newaxis, np.newaxis])
            net.params['rpn_bbox_pred'][1].data[...] = \
                (net.params['rpn_bbox_pred'][1].data *
                 self.rpn_stds + self.rpn_means)

        if scale_bbox_params_rfcn:
            # save original values
            orig_0 = net.params['rfcn_bbox'][0].data.copy()
            orig_1 = net.params['rfcn_bbox'][1].data.copy()
            repeat = orig_1.shape[0] / self.bbox_means.shape[0]

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['rfcn_bbox'][0].data[...] = \
                    (net.params['rfcn_bbox'][0].data *
                     np.repeat(self.bbox_stds, repeat).reshape((orig_1.shape[0], 1, 1, 1)))
            net.params['rfcn_bbox'][1].data[...] = \
                    (net.params['rfcn_bbox'][1].data *
                     np.repeat(self.bbox_stds, repeat) + np.repeat(self.bbox_means, repeat))

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)
        if self.gpu_id == 0:
            net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if scale_bbox_params_faster_rcnn:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1
        if scale_bbox_params_rfcn:
            # restore net to original state
            net.params['rfcn_bbox'][0].data[...] = orig_0
            net.params['rfcn_bbox'][1].data[...] = orig_1
        if scale_bbox_params_rpn:
            # restore net to original state
            net.params['rpn_bbox_pred'][0].data[...] = rpn_orig_0
            net.params['rpn_bbox_pred'][1].data[...] = rpn_orig_1

        return filename

    def getSolver(self):
        return self.solver

def solve(proto, pretrained_model, gpus, uid, rank, output_dir, max_iter):
    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)
    cfg.GPU_ID = gpus[rank]

    solverW = SolverWrapper(proto, output_dir, rank, pretrained_model)
    solver = solverW.getSolver()
    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()
    solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)
    count = 0
    while count < max_iter:
        solver.step(cfg.TRAIN.SNAPSHOT_ITERS)
        if rank == 0:
            solverW.snapshot()
        count = count + cfg.TRAIN.SNAPSHOT_ITERS

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = args.gpu_id
    gpu_list = gpu_id.split(',')
    gpus = [int(i) for i in gpu_list]

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        #caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe

    output_dir = args.out_dir
    print 'Output will be saved to `{:s}`'.format(output_dir)

    """Train a Fast R-CNN network."""
    uid = caffe.NCCL.new_uid()
    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))
    procs = []

    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(args.solver, args.pretrained_model, gpus, uid, rank, output_dir, args.max_iters))
        p.daemon = False
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
