from jacinle.utils.container import GView
from models.model import make_motion_reasoning_configs, MotionReasoningModel

import torch

configs = make_motion_reasoning_configs()

class Model(MotionReasoningModel):
    def __init__(self, no_gt_segments, temporal_operator, max_num_segments, args):
        self.no_gt_segments = no_gt_segments
        self.temporal_operator = temporal_operator
        self.max_num_segments = max_num_segments
        self.args = args
        super().__init__(configs)

    def forward(self, feed_dict):

        feed_dict = GView(feed_dict)
        monitors, outputs = {}, {}
        motion_encodings = self.motion_encoder(feed_dict.joints)

        # build scene
        f_sng = []
        start_seg = 0

        for seq_num_segs in feed_dict.num_segs:
            f_sng.append(motion_encodings[start_seg : start_seg + seq_num_segs])
            start_seg += seq_num_segs
        assert start_seg == motion_encodings.size()[0]

        feed_dict.f_sng = f_sng

        programs = feed_dict.program_qsseq

        logits = self.reasoning(feed_dict, self.args)

        # final loss
        update_from_loss_module(monitors, outputs, self.query_loss(feed_dict, logits))

        canonize_monitors(monitors)

        if self.training:
            if "loss/query" in monitors and "loss/filter" in monitors:
                loss = (
                    monitors["loss/query"] + monitors["loss/filter"]
                )  # can finetune ratio
            elif "loss/query" in monitors:
                loss = monitors["loss/query"]
            elif "loss/filter" in monitors:
                loss = monitors["loss/filter"]
            else:
                loss = 0  # happens when not using filter loss and all questions are filter related
            return loss, monitors, outputs
        else:
            outputs["monitors"] = monitors
            return outputs


def make_model(args, max_num_segments):
    return Model(args.no_gt_segments, args.temporal_operator, max_num_segments, args)


def canonize_monitors(monitors):
    for k, v in monitors.items():
        if isinstance(monitors[k], list):
            if isinstance(monitors[k][0], tuple) and len(monitors[k][0]) == 2:
                monitors[k] = sum([a * b for a, b in monitors[k]]) / max(
                    sum([b for _, b in monitors[k]]), 1e-6
                )
            else:
                monitors[k] = sum(v) / max(len(v), 1e-3)
        if isinstance(monitors[k], float):
            monitors[k] = torch.tensor(monitors[k])


def update_from_loss_module(monitors, output_dict, loss_update):
    # print("back in desc_nspose..")
    tmp_monitors, tmp_outputs = loss_update
    monitors.update(tmp_monitors)
    output_dict.update(tmp_outputs)
