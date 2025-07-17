import sys
import os
from os.path import join as pjoin
import math

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from motion_patch_clip import ClipModel
from transformers import RobertaTokenizer, RobertaModel

def getPositionEncoding(seq_len, d):
    n=10000
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)

    return P

def getGroupedPositionEncoding(seq_len, d, group_size):
    n = 10000
    num_groups = int(np.ceil(seq_len / group_size))  # Total number of groups
    P = torch.zeros((num_groups, d))  # Compute positional encodings for groups

    # Generate positional encodings for groups
    for k in range(num_groups):
        for i in range(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)

    # Assign positional encodings to the sequence
    position_embeddings = P.repeat_interleave(group_size, dim=0)[:seq_len]
    position_embeddings = position_embeddings.unsqueeze(0)

    return position_embeddings


class Conv1dTemporalProjection1(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_intermediate_layers = 3

        self.layer1 = nn.Conv1d(256, 1024, 3, padding=1)
        print(self.layer1)
        self.intermediate_layers = nn.ModuleList(
            [nn.Conv1d(1024, 1024, 3, padding=(3-1)*(2**(i + 1)) // 2, dilation=2**(i + 1)) for i in range(self.num_intermediate_layers)]
        ) 
        self.last_conv = nn.Conv1d(1024, 2048, 3, padding=1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = nn.ReLU()(x)

        for layer in self.intermediate_layers:
            x = nn.ReLU()(layer(x)) + x 

        x = self.last_conv(x)
        return x
    
class Conv1dTemporalProjection2(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_intermediate_layers = 3

        self.layer1 = nn.Conv1d(768, 1024, 3, padding=1) 
        self.last_conv = nn.Conv1d(1024, 2048, 3, padding=1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = nn.ReLU()(x)
        x = self.last_conv(x)
        return x
    
class Conv1dTemporalProjection3(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_intermediate_layers = 3
        self.layer1 = nn.Conv1d(256, 768, 3, padding=1) 
        self.last_conv = nn.Conv1d(768, 768, 3, padding=1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = nn.ReLU()(x)
        x = self.last_conv(x)
        return x


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

class MHAtt(nn.Module):
    def __init__(self, head_num, hidden_size, dropout, hidden_size_head):
        super(MHAtt, self).__init__()
        self.head_num = head_num
        self.hidden_size = 2048
        self.hidden_size_head = hidden_size_head
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        self.head_num = 8
        self.hidden_size_head = 64

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.head_num,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.head_num,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.head_num,
            self.hidden_size_head
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
    

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size, out_size)
        if use_relu:
            self.relu = nn.ReLU(inplace=False)
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r, inplace=False)

    def forward(self, x):
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        if self.dropout_r > 0:
            x = self.dropout(x)
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


### Feed Forward Nets
class FFN(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=ff_size,
            out_size=hidden_size,
            dropout_r=dropout,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


### Self Attention
class SA(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
        super(SA, self).__init__()

        self.mhatt = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
        self.ffn = FFN(hidden_size, ff_size, dropout)

        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, x_mask):
        output = self.mhatt(x, x, x, x_mask)
        dropout_output = self.dropout1(output)
        x = self.norm1(x + dropout_output)

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


### Self Guided Attention
class SGA(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
        self.mhatt2 = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
        self.ffn = FFN(hidden_size, ff_size, dropout)

        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(dropout, inplace=False)
        self.norm3 = LayerNorm(hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


class GA(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
        super(GA, self).__init__()

        self.mhatt = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
        self.ffn = FFN(hidden_size, ff_size, dropout)

        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, y, y_mask, x_mask=None):
        if x_mask is None:
            intermediate = self.dropout1(self.mhatt(y, y, x, y_mask))
        else:
            intermediate = self.dropout1(self.mhatt(y, y, x, y_mask)) * x_mask.unsqueeze(-1)

        x = self.norm1(x + intermediate)
        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x
    

class ShallowModule(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head, preprocessing=True):
        super(ShallowModule, self).__init__()
        self.cross_attention = GA(hidden_size, head_num, ff_size, dropout, hidden_size_head)

    def forward(self, inputs, motion_feat, mo_mask_tmp, program_masks):
        enc_output = self.cross_attention(inputs, motion_feat, mo_mask_tmp)
        enc_output = enc_output * program_masks.unsqueeze(-1)
        return enc_output


class IPGRM(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
        super(IPGRM, self).__init__()
        self.self_attention = SA(hidden_size, head_num, ff_size, dropout, hidden_size_head)
        self.cross_attention = GA(hidden_size, head_num, ff_size, dropout, hidden_size_head)

    def forward(self, inputs, mask, motion_feat, mo_mask_tmp, program_masks, alpha):
        alpha = alpha.unsqueeze(-1)
        trans_mask = (1 - mask).unsqueeze(1).to(torch.bool)
        enc_output = self.self_attention(inputs, trans_mask)
        enc_output = self.cross_attention(enc_output, motion_feat, mo_mask_tmp)
        enc_output = enc_output * program_masks.unsqueeze(-1)
        return alpha * enc_output + (1 - alpha) * inputs


### Load Motion ViT model
def prepare_test_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    motion_encoder_alias = 'vit_base_patch16_224_in21k'
    text_encoder_alias = 'distilbert-base-uncased'
    motion_embedding_dims: int = 768
    text_embedding_dims: int = 768
    projection_dims: int = 256

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_alias)

    model = ClipModel(
        motion_encoder_alias=motion_encoder_alias,
        text_encoder_alias=text_encoder_alias,
        motion_embedding_dims=motion_embedding_dims,
        text_embedding_dims=text_embedding_dims,
        projection_dims=projection_dims,
        patch_size=16,
    )

    model_path = pjoin('./pretrained_models/HumanML3D/', "best_model.pt")
    print(model_path)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.cuda()
    
    return model, tokenizer


class TreeTransformerSparsePostv2(nn.Module):
    def __init__(self, vocab_size, answer_size, visual_dim, hidden_dim, coordinate_dim,
                 n_head, n_layers, stacking, dropout, intermediate_dim, pre_layers, intermediate_layer):
        super(TreeTransformerSparsePostv2, self).__init__()

        PAD = 0
        self.embedding = nn.Embedding(vocab_size, 300, padding_idx=PAD)
        self.ques_proj = nn.Linear(768, hidden_dim)
        self.prog_proj = nn.Linear(300, hidden_dim // 8)
        self.qs_type_proj = nn.Linear(768, hidden_dim)

        self.ques_pos_emb = PositionalEmbedding(hidden_dim)
        self.intermediate_layer = intermediate_layer
   
        ### Motion projections
        self.motion_proj1 = Conv1dTemporalProjection1()
        self.motion_proj2 = Conv1dTemporalProjection2()
        self.motion_proj3 = Conv1dTemporalProjection3() 

        self.max_num_segments=5

        self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)

        ### Question encoder
        self.ques_encoder = nn.ModuleList(
            [SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
             for _ in range(pre_layers)])
        
        ### Motion decoder for attending question
        self.motion_encoder1 = nn.ModuleList(
            [SGA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
             for _ in range(pre_layers)])
        
        ### Motion decoder for attending question type
        self.motion_encoder2 = nn.ModuleList(
            [SGA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
             for _ in range(pre_layers)])

        ### IPGRM
        self.module = IPGRM(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)

        ### IPGRM post-processing decoder 
        self.post = nn.ModuleList([ShallowModule(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
                                    for _ in range(stacking)])
        
        ### IPGRM output decoder with question
        self.motion_encoder3 = nn.ModuleList(
            [SGA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
             for _ in range(pre_layers)])
        
        ### Question-type based classifiers
        self.action_classifier = nn.Linear(hidden_dim, 42)
        self.direction_classifier = nn.Linear(hidden_dim, 4)
        self.body_part_classifier = nn.Linear(hidden_dim, 8)

        ### Load RoBERTa model to get text embeddings
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.num_tokens=30

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.roberta_model.to(self.device)
        self.roberta_model.eval()

        ### Load Motion ViT (MotionPatches) model
        model, tokenizer = prepare_test_model()
        model.eval()
        self.motion_encoder = model

        ### Feature fusion in Motion ViT
        model, tokenizer = prepare_test_model()
        model.eval()
        self.motion_encoder = model

        # Accessing blocks in the VisionTransformer inside motion_encoder
        self.block_0 = self.motion_encoder.motion_encoder.model.blocks[0]
        self.block_2 = self.motion_encoder.motion_encoder.model.blocks[2]
        self.block_4 = self.motion_encoder.motion_encoder.model.blocks[4]
        self.block_6 = self.motion_encoder.motion_encoder.model.blocks[6]
        self.block_8 = self.motion_encoder.motion_encoder.model.blocks[8]
        self.block_11 = self.motion_encoder.motion_encoder.model.blocks[11]

        self.patch_embed = self.motion_encoder.motion_encoder.model.patch_embed

        self.out_indices = [0, 2, 4, 6, 8, 11]
        self.embed_dims = 768
        self.last_layer_dim = 256

        proj_layers = [
            torch.nn.Linear(self.embed_dims, 2048)
            for _ in range(len(self.out_indices))
        ]

        self.proj_layer_last = nn.Sequential(torch.nn.Linear(self.last_layer_dim, 1024), torch.nn.Linear(1024, 2048))
        self.proj_layers = torch.nn.ModuleList(proj_layers)
        self.norm1 = nn.LayerNorm(self.embed_dims, eps=1e-6)
    

    def get_intermediate_feat(self, motion_patches ):
            
        motion_patches= self.patch_embed(motion_patches)

        res = []
        motion_patches = self.block_0(motion_patches)
        motion_patches_proj = self.proj_layers[0](motion_patches)
        res.append(motion_patches_proj)
        motion_patches = self.block_2(motion_patches)
        motion_patches_proj = self.proj_layers[1](motion_patches)
        res.append(motion_patches_proj)
        motion_patches = self.block_4(motion_patches)
        motion_patches_proj = self.proj_layers[2](motion_patches)
        res.append(motion_patches_proj)
        motion_patches = self.block_6(motion_patches)
        motion_patches_proj = self.proj_layers[3](motion_patches)
        res.append(motion_patches_proj)
        motion_patches = self.block_8(motion_patches)
        motion_patches_proj = self.proj_layers[4](motion_patches)
        res.append(motion_patches_proj)
        motion_patches = self.block_11(motion_patches)
        motion_patches_proj = self.proj_layers[5](motion_patches)
        res.append(motion_patches_proj)

        motion_feat = torch.stack(res, dim=1)
        motion_feat = motion_feat[:, :, 0, :]

        return motion_feat
    

    def forward(self, feed_dict, args):

        ques = feed_dict.question
        batch_size = ques.size(0)

        ### Get motion embeddings from Motion ViT
        if args.featurefusion:
            motion_patches = feed_dict.motion_patches

            max_length = max(seq.shape[1] for seq in motion_patches)
            motion_patches = [F.pad(seq, (0, 0, 0, max_length - seq.shape[1])) for seq in motion_patches]
            motion_patches = torch.stack(motion_patches) # [4, 3, 75, 80] [B, C, T, J]

            max_motion_length = 224
            m_length = motion_patches.shape[2]

            if m_length <= max_motion_length:
                padding_len = max_motion_length - m_length
                B, C, T, J = motion_patches.shape
                padding_zeros = torch.zeros((B, C, padding_len, J), dtype=torch.float32).to(self.device)
                motion_patches = torch.cat((motion_patches, padding_zeros), dim=2)

                encoded_slices = []

                # Intermediate outputs from Motion ViT
                intermediate_feat = self.get_intermediate_feat(motion_patches)
                encoded_slices.append(intermediate_feat)

                # Motion ViT final encoder output
                final_feat = self.motion_encoder.encode_motion(motion_patches)
                final_feat = self.proj_layer_last(final_feat.unsqueeze(1))
                encoded_slices.append(final_feat)

                # Fusining Motion ViT final encoder output with intermediate outputs
                motion_feat = torch.cat(encoded_slices, dim=1)
                pos = getGroupedPositionEncoding(motion_feat.size(1), motion_feat.size(2), 7)
                motion_feat = motion_feat + pos.to(self.device)

            else:
                B, C, T, J = motion_patches.shape
                slice_size = 224

                # Pad the motion sequence to ensure T is divisible by slice_size
                padding_len = (slice_size - (T % slice_size)) % slice_size  
                if padding_len > 0:
                    padding = torch.zeros(B, C, padding_len, J, device=motion_patches.device)
                    motion_patches = torch.cat((motion_patches, padding), dim=2)

                # Slice the sequence, process with encoder, and concatenate the results
                encoded_slices = []
                for i in range(0, motion_patches.size(2), slice_size):
                    slice = motion_patches[:, :, i:i + slice_size, :]  # Extract slice along dim 2
                    
                    # Intermediate outputs from Motion ViT
                    intermediate_feat = self.get_intermediate_feat(slice)
                    encoded_slices.append(intermediate_feat)
                    
                    # Motion ViT final encoder output
                    final_feat = self.motion_encoder.encode_motion(slice)  # Pass slice through the Motion ViT encoder
                    final_feat = self.proj_layer_last(final_feat.unsqueeze(1))
                    encoded_slices.append(final_feat)

                # Fusining Motion ViT final encoder output with intermediate outputs
                motion_feat = torch.cat(encoded_slices, dim=1) # [[4, 6, 2048], [4, 1, 2048], ...]
                pos = getGroupedPositionEncoding(motion_feat.size(1), motion_feat.size(2), 7)
                motion_feat = motion_feat + pos.to(self.device)
        else:
            motion_patches = feed_dict.motion_patches
            max_length = max(seq.shape[1] for seq in motion_patches)
            motion_patches = [F.pad(seq, (0, 0, 0, max_length - seq.shape[1])) for seq in motion_patches]
            motion_patches = torch.stack(motion_patches) 

            max_motion_length = 224
            m_length = motion_patches.shape[2]

            if m_length <= max_motion_length:
                padding_len = max_motion_length - m_length
                B, C, T, J = motion_patches.shape
                padding_zeros = torch.zeros((B, C, padding_len, J), dtype=torch.float32).to(self.device)
                motion_patches = torch.cat((motion_patches, padding_zeros), dim=2)
                motion_feat = self.motion_encoder.encode_motion(motion_patches)
                motion_feat = motion_feat.unsqueeze(1)
            else:
                B, C, T, J = motion_patches.shape
                slice_size = 224

                padding_len = (slice_size - (T % slice_size)) % slice_size  
                if padding_len > 0:
                    padding = torch.zeros(B, C, padding_len, J, device=motion_patches.device)
                    motion_patches = torch.cat((motion_patches, padding), dim=2)

                encoded_slices = []
                for i in range(0, motion_patches.size(2), slice_size):
                    slice = motion_patches[:, :, i:i + slice_size, :]  # Extract slice along dim 2
                    encoded_slice = self.motion_encoder.encode_motion(slice)  # Pass slice through the Motion ViT encoder
                    encoded_slices.append(encoded_slice)

                encoded_slices = [s.unsqueeze(1) for s in encoded_slices]
                motion_feat = torch.cat(encoded_slices, dim=1)

            motion_feat = motion_feat / motion_feat.norm(dim=2, keepdim=True)

            X, Y, Z = motion_feat.shape
            motion_feat = self.motion_proj1(motion_feat.view(X, Z, Y))
            X, Z, Y = motion_feat.shape
            motion_feat = motion_feat.view(X, Y, Z)

        ### Question type embeddings and masks
        qs_type_embeds = []
        qs_type_list = feed_dict.query_type
        for i in range (len(qs_type_list)):
            qs_type_encoding = self.tokenizer.encode_plus(
                qs_type_list[i],
                add_special_tokens=True, 
                max_length=self.num_tokens,
                return_token_type_ids=False,
                padding='max_length', 
                truncation=True,  
                return_attention_mask=True,
                return_tensors='pt'
            )

            for key in qs_type_encoding:
                qs_type_encoding[key] = qs_type_encoding[key].to(self.device)

            outputs = self.roberta_model(**qs_type_encoding)

            last_hidden_state = outputs.last_hidden_state 
            emb = last_hidden_state.squeeze(0)
            qs_type_embeds.append(emb)

        qs_type = torch.stack(qs_type_embeds)
        qs_type_input = self.qs_type_proj(qs_type)
        qs_type_mask_tmp = (1 - feed_dict.qs_type_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)

        ### Question embeddings and masks
        ques_embeds = []
        ques_list = feed_dict.question_text
        for i in range (len(ques_list)):
            encoding = self.tokenizer.encode_plus(
                ques_list[i],
                add_special_tokens=True,  
                max_length=self.num_tokens,
                return_token_type_ids=False,
                padding='max_length',  
                truncation=True,  
                return_attention_mask=True,
                return_tensors='pt'
            )

            for key in encoding:
                encoding[key] = encoding[key].to(self.device)

            outputs = self.roberta_model(**encoding)

            last_hidden_state = outputs.last_hidden_state  # (1, max_len, hidden_size)
            emb = last_hidden_state.squeeze(0)
            ques_embeds.append(emb)

        ques = torch.stack(ques_embeds)

        ques_masks = feed_dict.question_masks
        ques_mask_tmp = (1 - ques_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)
        ques_input = self.ques_proj(ques) + self.ques_pos_emb(ques)	

        for enc in self.ques_encoder:
            ques_input = enc(ques_input, ques_mask_tmp)
            ques_input *= ques_masks.unsqueeze(-1)

        mo_mask_tmp = None

        ### Motion embeddings attend question
        for enc in self.motion_encoder1:
            motion_feat = enc(motion_feat, ques_input, mo_mask_tmp, ques_mask_tmp)

        ### Motion embeddings attend question type
        for enc in self.motion_encoder2:
            motion_feat = enc(motion_feat, qs_type_input , mo_mask_tmp, qs_type_mask_tmp)

        ### Encoding the program
        program = feed_dict.program
        enc_output = self.prog_proj(self.embedding(program)).view(batch_size, program.size(1), -1)
        transition_masks = feed_dict.transition_masks.transpose(0, 1)
        activate_masks = feed_dict.activate_mask.transpose(0, 1)

        ### IPGRM: Implicit Program-Guided Reasoning Module
        for trans_mask, active_mask in zip(transition_masks, activate_masks):
            enc_output = self.module(enc_output, trans_mask, motion_feat, mo_mask_tmp, feed_dict.program_masks, active_mask)

        ### IPGRM's output attend question
        for enc in self.motion_encoder3:
            enc_output = enc(enc_output, ques_input, mo_mask_tmp, ques_mask_tmp)

        ### Post-Processing the encoder output
        for layer in self.post:
            enc_output = layer(enc_output, motion_feat, mo_mask_tmp, feed_dict.program_masks)

        lang_feat = torch.gather(enc_output, 1, feed_dict.index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, enc_output.size(-1)))
        lang_feat = lang_feat.view(batch_size, -1)

        ### Question type-based classifiers
        logits = []
        for i in range(len(qs_type_list)):
            if qs_type_list[i] == 'query_body_part':
                l = self.body_part_classifier(lang_feat[i])
                logits.append(l)
            elif qs_type_list[i] == 'query_action':
                l = self.action_classifier(lang_feat[i])
                logits.append(l)
            elif qs_type_list[i] == 'query_direction':
                l = self.direction_classifier(lang_feat[i])
                logits.append(l)
            else:
                print("Unknown query category!!")

        return logits
