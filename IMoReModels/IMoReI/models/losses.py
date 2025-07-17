import torch.nn as nn
import torch
import torch.nn.functional as F

class MultitaskLossBase(nn.Module):
    def __init__(self):
        super().__init__()

    def _xent_loss(self, pred, label):
        logp = F.log_softmax(pred, dim=-1)
        return -logp[label].mean()

    def cross_entropy_loss(self, pred, label):
        return F.cross_entropy(pred, label)


class IntermediateFilterMonitor(MultitaskLossBase):
    def __init__(self):
        super().__init__()

    def forward(self, feed_dict, answers, buffers):
        monitors = {}
        outputs = {}

        for i, prog in enumerate(feed_dict.program_qsseq):
            filter_num = 0
            for block_id, block in enumerate(prog):
                if block["op"] == "filter":
                    inter_a = buffers[i][block_id]
                    argmax = inter_a.argmax(dim=-1).item()

                    # change to middle of argmax segment intersects with gt
                    middle_seg = (
                        feed_dict["segment_boundaries"][i][argmax][0]
                        + feed_dict["segment_boundaries"][i][argmax][1]
                    ) / 2
                    acc = (
                        middle_seg >= feed_dict["filter_boundaries"][i][filter_num][0]
                        and middle_seg
                        <= feed_dict["filter_boundaries"][i][filter_num][1]
                    )

                    monitors.setdefault(
                        feed_dict.program_raw[i][block_id]["function"], []
                    ).append((acc, 1))
                    filter_num += 1

        return monitors, outputs


class QueryLoss(MultitaskLossBase):
    def __init__(self):
        super().__init__()

    def forward(self, feed_dict, logits):

        monitors = {}
        outputs = {"answer": [], "gt": []}

        gt = feed_dict.answer_id

        pred = []
        predicted_list= []
        for i in range(len(logits)):

            predicted = torch.argmax(logits[i], -1)
            predicted_list.append(predicted)

            outputs["answer"].append(predicted)
            outputs["gt"].append(gt[i])
            pred.append(predicted)
            success_or_not = (predicted == gt[i]).float()
            monitors.setdefault("acc/qa", []).append((int(predicted == gt[i]), 1))

            query_type = feed_dict["query_type"][i]
            monitors.setdefault("acc/" + query_type, []).append(
                (int(predicted == gt[i]), 1)
            )

            relation_type = feed_dict.relation_type[i]
            query_type_specific = f"{query_type}_{relation_type}"
            monitors.setdefault("acc/" + query_type_specific, []).append(
                (int(predicted == gt[i]), 1)
            )

        feed_dict.pred_answer_id = predicted_list

        ce_loss = self.cross_entropy_loss
        if self.training:
            for i in range(len(logits)):
                l = ce_loss(logits[i], gt[i])
                monitors.setdefault("loss/query", []).append((l, 1))

        return monitors, outputs
