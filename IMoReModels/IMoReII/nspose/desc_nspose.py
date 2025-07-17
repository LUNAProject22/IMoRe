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

    def find_best_run(self, logits_list):
        """
        Finds the run with the highest overall logits across all batches and questions.

        logits_list: List of 5 runs, each containing 4 batches, where each batch has 4 logits tensors.

        Returns:
        - run with the best_run_index (int): The index of the run with the highest overall logits.
        - highest_logit_value (float): The sum of all logits in the best run.
        """
        total_logits_per_run = []  # Store total logits per run

        for run in logits_list:
            run_total_logit = 0  # Sum of logits for the run

            for batch in run:
                for logits in batch:
                    run_total_logit += logits.sum().item()  # Sum all logits in this batch

            total_logits_per_run.append(run_total_logit)

        # Find the run with the highest overall logit sum
        best_run_index = torch.tensor(total_logits_per_run).argmax().item()
        highest_logit_value = max(total_logits_per_run)

        return logits_list[best_run_index]


    # #### Mean
    # def find_best_run(self, logits_list):
    #     """
    #     Finds the run with the highest mean logits across all batches and questions.

    #     logits_list: List of 5 runs, each containing 4 batches, where each batch has 4 logits tensors.

    #     Returns:
    #     - run with the best_run_index (int): The index of the run with the highest mean logits.
    #     - highest_logit_value (float): The mean of all logits in the best run.
    #     """
    #     mean_logits_per_run = []  # Store mean logits per run

    #     for run in logits_list:
    #         run_total_logit = 0  # Sum of logits for the run
    #         count = 0  # Count of all logits in the run

    #         for batch in run:
    #             for logits in batch:
    #                 run_total_logit += logits.sum().item()  # Sum all logits in this batch
    #                 count += 1 # Count total elements

    #         mean_logit = run_total_logit / count if count > 0 else 0  # Compute mean
    #         mean_logits_per_run.append(mean_logit)

    #     # Find the run with the highest mean logit
    #     best_run_index = torch.tensor(mean_logits_per_run).argmax().item()
    #     highest_logit_value = max(mean_logits_per_run)

    #     return logits_list[best_run_index]


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

        logits_list = []
        if not self.training: # default should be 'not self.training'
            for i in range(5):
                logits = self.reasoning(feed_dict, self.args)
                logits_list.append(logits)
        
            stacked_tensors  = list(zip(*logits_list))
            # mean_logits = [torch.stack(tensors, dim=0).mean(dim=0) for tensors in stacked_tensors ]

            mean_logits = self.find_best_run(logits_list)

        else:
            mean_logits = self.reasoning(feed_dict, self.args) 


        # final loss
        update_from_loss_module(monitors, outputs, self.query_loss(feed_dict, mean_logits))

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
