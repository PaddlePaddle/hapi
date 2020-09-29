# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.static import InputSpec
from paddle.utils.download import get_weights_path_from_url

from darknet import darknet53

__all__ = ['YoloLoss', 'YOLOv3', 'yolov3_darknet53']

# {depth: (url, md5)}
pretrain_infos = {
    53: ('https://paddlemodels.bj.bcebos.com/hapi/yolov3_darknet53.pdparams',
         '809a07f541b7caf11d0c806986643c0f')
}


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act="leaky"):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)
        self.batch_norm = nn.BatchNorm2d(
            ch_out,
            weight_attr=ParamAttr(regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.)))

        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = F.leaky_relu(out, 0.1)
        return out


class YoloDetectionBlock(nn.Layer):
    def __init__(self, ch_in, channel):
        super(YoloDetectionBlock, self).__init__()

        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2".format(channel)

        self.conv0 = ConvBNLayer(
            ch_in=ch_in, ch_out=channel, filter_size=1, stride=1, padding=0)
        self.conv1 = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            filter_size=3,
            stride=1,
            padding=1)
        self.conv2 = ConvBNLayer(
            ch_in=channel * 2,
            ch_out=channel,
            filter_size=1,
            stride=1,
            padding=0)
        self.conv3 = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            filter_size=3,
            stride=1,
            padding=1)
        self.route = ConvBNLayer(
            ch_in=channel * 2,
            ch_out=channel,
            filter_size=1,
            stride=1,
            padding=0)
        self.tip = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            filter_size=3,
            stride=1,
            padding=1)

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return route, tip


class YOLOv3(nn.Layer):
    """YOLOv3 model from
    `"YOLOv3: An Incremental Improvement" <https://arxiv.org/abs/1804.02767>`_

    Args:
        num_classes (int): class number, default 80.
        model_mode (str): 'train', 'eval', 'test' mode, network structure
            will be diffrent in the output layer and data, in 'train' mode,
            no output layer append, in 'eval' and 'test', output feature
            map will be decode to predictions by 'paddle.nn.functional.yolo_box',
            in 'eval' mode, return feature maps and predictions, in 'test'
            mode, only return predictions. Default 'train'.

    """

    def __init__(self, num_classes=80, model_mode='train'):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        assert str.lower(model_mode) in ['train', 'eval', 'test'], \
            "model_mode should be 'train' 'eval' or 'test', but got " \
            "{}".format(model_mode)
        self.model_mode = str.lower(model_mode)
        self.anchors = [
            10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198,
            373, 326
        ]
        self.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.valid_thresh = 0.005
        self.nms_thresh = 0.45
        self.nms_topk = 400
        self.nms_posk = 100
        self.draw_thresh = 0.5

        self.backbone = darknet53(pretrained=(model_mode == 'train'))
        self.block_outputs = []
        self.yolo_blocks = []
        self.route_blocks = []

        for idx, num_chan in enumerate([1024, 768, 384]):
            yolo_block = self.add_sublayer(
                "yolo_detecton_block_{}".format(idx),
                YoloDetectionBlock(num_chan, 512 // (2**idx)))
            self.yolo_blocks.append(yolo_block)

            num_filters = len(self.anchor_masks[idx]) * (self.num_classes + 5)

            block_out = self.add_sublayer(
                "block_out_{}".format(idx),
                nn.Conv2d(
                    in_channels=1024 // (2**idx),
                    out_channels=num_filters,
                    kernel_size=1,
                    weight_attr=ParamAttr(
                        initializer=nn.initializer.Normal(0., 0.02)),
                    bias_attr=ParamAttr(
                        initializer=nn.initializer.Constant(0.0),
                        regularizer=L2Decay(0.))))
            self.block_outputs.append(block_out)
            if idx < 2:
                route = self.add_sublayer(
                    "route2_{}".format(idx),
                    ConvBNLayer(
                        ch_in=512 // (2**idx),
                        ch_out=256 // (2**idx),
                        filter_size=1,
                        act='leaky_relu'))
                self.route_blocks.append(route)

    def forward(self, img_id, img_shape, inputs):
        outputs = []
        boxes = []
        scores = []
        downsample = 32

        feats = self.backbone(inputs)
        route = None
        for idx, feat in enumerate(feats[::-1]):
            if idx > 0:
                feat = paddle.concat(x=[route, feat], axis=1)
            route, tip = self.yolo_blocks[idx](feat)
            block_out = self.block_outputs[idx](tip)
            outputs.append(block_out)

            if idx < 2:
                route = self.route_blocks[idx](route)
                route = F.resize_nearest(route, scale=2)

            if self.model_mode != 'train':
                anchor_mask = self.anchor_masks[idx]
                mask_anchors = []
                for m in anchor_mask:
                    mask_anchors.append(self.anchors[2 * m])
                    mask_anchors.append(self.anchors[2 * m + 1])
                b, s = F.yolo_box(
                    x=block_out,
                    img_size=img_shape,
                    anchors=mask_anchors,
                    class_num=self.num_classes,
                    conf_thresh=self.valid_thresh,
                    downsample_ratio=downsample)

                boxes.append(b)
                scores.append(paddle.transpose(s, perm=[0, 2, 1]))

            downsample //= 2

        if self.model_mode == 'train':
            return outputs

        preds = [
            img_id, F.multiclass_nms(
                bboxes=paddle.concat(
                    boxes, axis=1),
                scores=paddle.concat(
                    scores, axis=2),
                score_threshold=self.valid_thresh,
                nms_top_k=self.nms_topk,
                keep_top_k=self.nms_posk,
                nms_threshold=self.nms_thresh,
                background_label=-1)
        ]

        if self.model_mode == 'test':
            return preds

        # model_mode == "eval"
        return outputs + preds


class YoloLoss(nn.Layer):
    def __init__(self, num_classes=80, num_max_boxes=50):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.num_max_boxes = num_max_boxes
        self.ignore_thresh = 0.7
        self.anchors = [
            10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198,
            373, 326
        ]
        self.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    def forward(self, *inputs):
        downsample = 32
        losses = []

        # YOLOv3 output fields is different between 'train' and 'eval' mode
        if len(inputs) == 6:
            output1, output2, output3, gt_box, gt_label, gt_score = inputs
        elif len(inputs) == 8:
            output1, output2, output3, img_id, bbox, gt_box, gt_label, gt_score = inputs

        outputs = [output1, output2, output3]
        for idx, out in enumerate(outputs):
            if idx == 3: break  # debug
            anchor_mask = self.anchor_masks[idx]
            loss = F.yolov3_loss(
                x=out,
                gt_box=gt_box,
                gt_label=gt_label,
                gt_score=gt_score,
                anchor_mask=anchor_mask,
                downsample_ratio=downsample,
                anchors=self.anchors,
                class_num=self.num_classes,
                ignore_thresh=self.ignore_thresh,
                use_label_smooth=False)
            loss = paddle.reduce_mean(loss)
            losses.append(loss)
            downsample //= 2
        return losses


def _yolov3_darknet(num_layers=53,
                    num_classes=80,
                    num_max_boxes=50,
                    model_mode='train',
                    pretrained=True):
    inputs = [
        InputSpec(
            [None, 1], 'int64', name='img_id'), InputSpec(
                [None, 2], 'int32', name='img_shape'), InputSpec(
                    [None, 3, None, None], 'float32', name='image')
    ]
    labels = [
        InputSpec(
            [None, num_max_boxes, 4], 'float32', name='gt_bbox'), InputSpec(
                [None, num_max_boxes], 'int32', name='gt_label'), InputSpec(
                    [None, num_max_boxes], 'float32', name='gt_score')
    ]
    net = YOLOv3(num_classes, model_mode)
    model = paddle.Model(net, inputs, labels)
    if pretrained:
        assert num_layers in pretrain_infos.keys(), \
                "YOLOv3-DarkNet{} do not have pretrained weights now, " \
                "pretrained should be set as False".format(num_layers)
        weight_path = get_weights_path_from_url(*(pretrain_infos[num_layers]))
        assert weight_path.endswith('.pdparams'), \
                "suffix of weight must be .pdparams"
        model.load(weight_path)
    return model


def yolov3_darknet53(num_classes=80,
                     num_max_boxes=50,
                     model_mode='train',
                     pretrained=True):
    """YOLOv3 model with 53-layer DarkNet as backbone
    
    Args:
        num_classes (int): class number, default 80.
        num_classes (int): max bbox number in a image, default 50.
        model_mode (str): 'train', 'eval', 'test' mode, network structure
            will be diffrent in the output layer and data, in 'train' mode,
            no output layer append, in 'eval' and 'test', output feature
            map will be decode to predictions by 'paddle.nn.functional.yolo_box',
            in 'eval' mode, return feature maps and predictions, in 'test'
            mode, only return predictions. Default 'train'.
        pretrained (bool): If True, returns a model with pre-trained model
            on COCO, default True
    """
    return _yolov3_darknet(53, num_classes, num_max_boxes, model_mode,
                           pretrained)

