
import jactorch.nn as jacnn
from jacinle.logging import get_logger
from jacinle.utils.container import G

import json
import torch.nn as nn


from datasets.definition import gdef

logger = get_logger(__file__)

__all__ = ['make_motion_reasoning_configs', 'MotionReasoningModel']


class Config(G):
    pass

def make_base_configs():
    configs = Config()

    configs.data = G()
    configs.model = G()
    configs.train = G()
    configs.train.weight_decay = 1e-4

    return configs

def make_motion_reasoning_configs():
    configs = make_base_configs()
    configs.model.sg_dims = [None, 256, 256]
    configs.model.vse_known_belong = False
    configs.model.vse_large_scale = False
    configs.model.vse_hidden_dims = [None, 64, 64]

    with open('./IPGRM_formatted_data/BABELQA_full_vocab.json', 'r') as f:
        vocab = json.load(f)
        ivocab = {v: k for k, v in vocab.items()}

    with open('./IPGRM_formatted_data/BABELQA_answer_vocab.json', 'r') as f:
        answer= json.load(f)

    configs.model.vocab_size=len(vocab)
    configs.stacking=6 
    configs.answer_size=len(answer)
    configs.visual_dim=2048
    configs.coordinate_dim=6
    configs.hidden_dim=512
    configs.n_head=8 
    configs.n_layers=5
    configs.dropout=0.1 
    configs.intermediate_dim=48 + 1 
    configs.pre_layers=3
    configs.intermediate_layer=False

    return configs


class MotionReasoningModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        from models.agcn import Model as AGCNModel
        self.motion_encoder = AGCNModel(False)

        import models.IPGRM_reasoning as mmn
        self.reasoning = mmn.TreeTransformerSparsePostv2(

            configs.model.vocab_size,
            configs.stacking, 
            configs.answer_size,
            configs.visual_dim,
            configs.coordinate_dim, 
            configs.hidden_dim, 
            configs.n_head, 
            configs.n_layers,
            configs.dropout, 
            configs.intermediate_dim, 
            configs.pre_layers,
            configs.intermediate_layer,
                )

        import models.losses as vqalosses
        self.query_loss = vqalosses.QueryLoss()

    def train(self, mode=True):
        super().train(mode)

    def _make_vse_concepts(self, large_scale, known_belong):
        return {
            'attribute': {
                'attributes': list(gdef.attribute_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.attribute_concepts.items() for v in vs
                ]
            },
            'relation': {
                'attributes': list(gdef.relational_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.relational_concepts.items() for v in vs
                ]
            }
        }