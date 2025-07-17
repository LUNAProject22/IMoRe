import os.path as osp
import math
import json

import numpy as np
import torch
import cv2

import jacinle.io as io
from jacinle.logging import get_logger
from jacinle.utils.container import GView
from jactorch.data.dataset import FilterableDatasetUnwrapped, FilterableDatasetView
from nltk.tokenize import word_tokenize

from datasets.humanmotionqa.utils import nsclseq_to_nscltree, nsclseq_to_nsclqsseq, nscltree_to_nsclqstree, program_to_nsclseq


logger = get_logger(__file__)

__all__ = ['NSTrajDataset']
        

class NSTrajDatasetUnwrapped(FilterableDatasetUnwrapped):
    def __init__(self, data_dir, data_split_file, split, downsample, no_gt_segments, num_frames_per_seg, overlapping_frames, max_frames=150):
        super().__init__()

        self.labels_json = osp.join(data_dir, 'motion_concepts.json')
        self.questions_json = osp.join(data_dir, 'questions.json')
        self.joints_root = osp.join(data_dir, 'motion_sequences')
        
        self.labels = io.load_json(self.labels_json)
        self.questions = io.load_json(self.questions_json)
        self.split_question_ids = io.load_json(data_split_file)[split]

        self.max_frames = max_frames
        self.no_gt_segments = no_gt_segments
        self.num_frames_per_seg = num_frames_per_seg
        self.overlapping_frames = overlapping_frames
        self.downsample = downsample

        self.mode = split
        
        with open('./IPGRM_formatted_data/BABELQA_{}_inputs.json'.format(self.mode), 'r') as f:
            self.data = json.load(f)
        print("loading data from {}".format('questions/BABELQA_{}_inputs.json'.format(self.mode)))

        print("there are in total {} instances".format(len(self.data)))

        with open('./IPGRM_formatted_data/BABELQA_full_vocab.json', 'r') as f:
            vocab = json.load(f)
            ivocab = {v: k for k, v in vocab.items()}

        with open('./IPGRM_formatted_data/BABELQA_answer_vocab.json', 'r') as f:
            answer= json.load(f)
            inv_answer = {v: k for k, v in answer.items()} 

        with open('./IPGRM_formatted_data/BABELQA_action_answer_vocab.json', 'r') as f:
            self.action_answer= json.load(f)

        with open('./IPGRM_formatted_data/BABELQA_body_part_answer_vocab.json', 'r') as f:
            self.body_part_answer= json.load(f)

        with open('./IPGRM_formatted_data/BABELQA_direction_answer_vocab.json', 'r') as f:
            self.direction_answer= json.load(f)

        self.vocab = vocab
        self.answer_vocab = answer
        self.num_tokens = 30
        self.num_regions = 48
        self.LENGTH =9
        self.MAX_LAYER =  7 #5

        self.threshold = 0.
        self.contained_weight = 0.1
        self.cutoff = 0.5
        self.distribution = False
    
    def _get_metainfo(self, index):
        question = self.questions[self.split_question_ids[index]]

        # program section
        has_program = False
        if 'program_nsclseq' in question:
            question['program_raw'] = question['program_nsclseq']
            question['program_seq'] = question['program_nsclseq']
            has_program = True
        elif 'program' in question:
            question['program_raw'] = question['program']
            question['program_seq'] = program_to_nsclseq(question['program'])
            has_program = True

        if has_program:
            question['program_tree'] = nsclseq_to_nscltree(question['program_seq'])
            question['program_qsseq'] = nsclseq_to_nsclqsseq(question['program_seq'])
            question['program_qstree'] = nscltree_to_nsclqstree(question['program_tree'])

        return question

    def sample_motion_sequence(self, sequence, sample_factor=4):
        """
        Downsamples the motion sequence along the time (T) dimension by a given factor.

        Args:
            sequence (numpy.ndarray): Input motion sequence of shape (T, V, C).
            sample_factor (int): The downsampling factor. Default is 4.

        Returns:
            numpy.ndarray: Downsampled motion sequence.
        """
        if len(sequence.shape) != 3:
            raise ValueError("Input sequence must have shape (T, V, C).")
        
        # Downsample by taking every nth frame
        downsampled_sequence = sequence[::sample_factor, :, :]
        return downsampled_sequence
    
    def __getitem__(self, index):
        metainfo = GView(self.get_metainfo(index))
        feed_dict = GView()

        question = self.questions[self.split_question_ids[index]]

        if 'program_raw' in metainfo:
            feed_dict.program_raw = metainfo.program_raw
            feed_dict.program_seq = metainfo.program_seq
            feed_dict.program_tree = metainfo.program_tree
            feed_dict.program_qsseq = metainfo.program_qsseq
            feed_dict.program_qstree = metainfo.program_qstree

        feed_dict.answer = question['answer']
        feed_dict.query_type = question['query_type']
        feed_dict.relation_type = question['relation_type']
        feed_dict.segment_boundaries = []
        feed_dict.question_text = question['question']
        feed_dict.question_id = question['question_id']

        # process joints
        id_name = 'babel_id'
        motion_id = question[id_name]
        num_segments = len(self.labels[motion_id])
        feed_dict.babel_id = motion_id

        joints = np.load(osp.join(self.joints_root, motion_id, 'joints.npy')) # T, V, C
        # change shape of joints to match model
        joints = joints[:, :, :, np.newaxis] # T, V, C, M
        joints = joints.transpose(2, 0, 1, 3) # C, T, V, M

        # label info
        labels_frame_info = self.labels[motion_id]

        if 'filter_answer_0' in question:
            filter_segment = labels_frame_info[question['filter_answer_0']]
            if filter_segment['end_f'] > np.shape(joints)[1]:
                filter_segment['end_f'] = np.shape(joints)[1]
            feed_dict.filter_boundaries = [(filter_segment['start_f'], filter_segment['end_f'])]
            if 'filter_answer_1' in question:
                filter_segment = labels_frame_info[question['filter_answer_1']]
                if filter_segment['end_f'] > np.shape(joints)[1]:
                    filter_segment['end_f'] = np.shape(joints)[1]
                feed_dict.filter_boundaries.append((filter_segment['start_f'], filter_segment['end_f']))

        if not self.no_gt_segments:
            joints_combined = np.zeros((num_segments, 3, self.max_frames, 22, 1), dtype=np.float32) # num_segs, C, T, V, M

            for seg_i, seg in enumerate(labels_frame_info):
                if seg['end_f'] > np.shape(joints)[1]: # end frame can be slightly off
                    seg['end_f'] = np.shape(joints)[1]
                num_frames = seg['end_f'] - seg['start_f']
                
                if num_frames > self.max_frames: # clip segments to max_frames
                    num_frames = self.max_frames
                
                joints_combined[seg_i, :, :num_frames, :, :] = joints[:, seg['start_f']: seg['start_f'] + num_frames, :, :]

                feed_dict.segment_boundaries.append((seg['start_f'], (seg['start_f'] + num_frames)))
            
            feed_dict.joints = joints_combined
            feed_dict.num_segs = num_segments
        else:
            total_num_frames = np.shape(joints)[1]
            num_segments = math.ceil(total_num_frames / self.num_frames_per_seg)
            feed_dict['info'] = []
            joints_combined = np.zeros((num_segments, 3, self.num_frames_per_seg + self.overlapping_frames*2, 22, 1), dtype=np.float32) # num_segs, C, T, V, M
            for i in range(num_segments):
                start_f = i * self.num_frames_per_seg
                end_f = (i + 1) * self.num_frames_per_seg
                if end_f > total_num_frames: end_f = total_num_frames

                missing_before_context = self.overlapping_frames - start_f if start_f < self.overlapping_frames else 0
                existing_after_context = total_num_frames - end_f
                if existing_after_context > self.overlapping_frames: existing_after_context = self.overlapping_frames
                
                joints_combined[i, :, missing_before_context:self.overlapping_frames+(end_f - start_f)+existing_after_context, :, :] = joints[:, start_f - (self.overlapping_frames - missing_before_context):end_f + existing_after_context, :, :]

                feed_dict.segment_boundaries.append((start_f - (self.overlapping_frames - missing_before_context), end_f + existing_after_context))
            
            feed_dict.joints = joints_combined
            feed_dict.num_segs = num_segments

        
        entry = self.data[index]
        if not entry[0].startswith('n'):
            if len(entry[0]) < 7:
                entry[0] = "0" * (7 - len(entry[0])) + entry[0]

        question = entry[1]
        inputs = entry[3]
        con = entry[4]
        connection = [[item] for item in con]
        questionId = entry[-2]

        length = min(len(inputs), self.LENGTH)

        # Prepare Question
        UNK = 3
        PAD = 1

        from transformers import RobertaTokenizer, RobertaModel

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        # Tokenizing the question using RoBERTa tokenizer
        encoding = tokenizer.encode_plus(
            question,
            add_special_tokens=True,  # Add special tokens like [CLS] and [SEP]
            max_length=self.num_tokens,
            return_token_type_ids=False,
            padding='max_length',  # Pad to max length
            truncation=True,  # Truncate if the sequence is longer than max length
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten().detach()
        question = input_ids.tolist()
        question = np.array(question, 'int64')

        # Prepare Question type and mask
        encoding_qs_type = tokenizer.encode_plus(
            feed_dict.query_type,
            add_special_tokens=True,  # Add special tokens like [CLS] and [SEP]
            max_length=self.num_tokens,
            return_token_type_ids=False,
            padding='max_length',  # Pad to max length
            truncation=True,  # Truncate if the sequence is longer than max length
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids =encoding_qs_type['input_ids'].flatten().detach()
        qs_type = input_ids.tolist()
        qs_type = np.array(qs_type, 'int64')
        
        question_type_masks = np.zeros((self.LENGTH, ), 'float32')
        question_type_masks[:length] = 1.

        question_masks = np.zeros((len(question), ), 'float32')
        question_masks[:length] = 1.
        
        program = np.zeros((self.LENGTH, 8), 'int64')
        depth = np.zeros((self.LENGTH, ), 'int64')
        for i in range(length):
            for j, text in enumerate(inputs[i]):
                if text is not None:
                    program[i][j] = self.vocab.get(text, UNK)

        # Prepare Program mask
        program_masks = np.zeros((self.LENGTH, ), 'float32')
        program_masks[:length] = 1.
        
        # Prepare Program Transition Mask
        transition_masks = np.zeros(
            (self.MAX_LAYER, self.LENGTH, self.LENGTH), 'uint8')
        activate_mask = np.zeros((self.MAX_LAYER, self.LENGTH), 'float32')
        for i in range(self.MAX_LAYER):
            if i < len(connection):
                for idx, idy in connection[i]:
                    transition_masks[i][idx][idy] = 1
                    depth[idx] = i
                    activate_mask[i][idx] = 1
            for j in range(self.LENGTH):
                if activate_mask[i][j] == 0:
                    # As a placeholder
                    transition_masks[i][j][j] = 1
                else:
                    pass
        
        vis_mask = np.zeros((self.num_regions, ), 'float32')

        # Prepare index selection
        index = length - 1

        # query type based GT
        if feed_dict.query_type == 'query_body_part':
            answer_id = self.body_part_answer.get(entry[-1], UNK)
        elif feed_dict.query_type == 'query_action':
            answer_id = self.action_answer.get(entry[-1], UNK)
        elif feed_dict.query_type == 'query_direction':
            answer_id = self.direction_answer.get(entry[-1], UNK)
        else:
            print("Unknown query category!!")

        object_feat = torch.rand(2,5)
        bbox_feat = torch.rand(2,5)
        intermediate_idx = torch.rand(2,5)

        feed_dict.length = length
        feed_dict.qs_type = qs_type
        feed_dict.qs_type_masks = question_type_masks
        feed_dict.question = question
        feed_dict.question_masks = question_masks
        feed_dict.program = program
        feed_dict.program_masks = program_masks
        feed_dict.transition_masks = transition_masks
        feed_dict.activate_mask = activate_mask
        feed_dict.object_feat = object_feat
        feed_dict.bbox_feat = bbox_feat
        feed_dict.vis_mask = vis_mask
        feed_dict.index = index
        feed_dict.depth = depth
        feed_dict.intermediate_idx = intermediate_idx
        feed_dict.answer_id = answer_id
        feed_dict.questionId = questionId

        # process the motion seq for motion_patches encoder
        self.kinematic_chain = self.kinematic_chain = [
                [0, 2, 5, 8, 11],
                [0, 1, 4, 7, 10],
                [0, 3, 6, 9, 12, 15],
                [9, 14, 17, 19, 21],
                [9, 13, 16, 18, 20],
            ]

        self.patch_size = 16

        def use_kinematic(motion):
            if self.patch_size == 16:
                motion_ = np.zeros(
                    (motion.shape[0], len(self.kinematic_chain) * 16, motion.shape[2]),
                    float,
                )
                for i_frames in range(motion.shape[0]):
                    for i, kinematic_chain in enumerate(self.kinematic_chain):
                        joint_parts = motion[i_frames, kinematic_chain]
                        joint_parts = joint_parts.reshape(1, -1, 3)
                        joint_parts = cv2.resize(
                            joint_parts, (16, 1), interpolation=cv2.INTER_LINEAR
                        )
                        motion_[i_frames, 16 * i : 16 * (i + 1)] = joint_parts[0]

            else:
                raise NotImplementedError

            return motion_

        raw_joints = np.load(osp.join(self.joints_root, motion_id, 'joints.npy')) # T, V, C
        if self.downsample > 1:
            raw_joints = self.sample_motion_sequence(raw_joints, sample_factor=self.downsample)

        self.mean = np.load('./IPGRM_formatted_data/BABEL_QA_mean.npy') # Mean_raw.npy 
        self.std = np.load('./IPGRM_formatted_data/BABEL_QA_std.npy') # Std_raw.npy

        raw_joints = (raw_joints - self.mean[np.newaxis, ...]) / self.std[np.newaxis, ...] 

        motion_patches = use_kinematic(raw_joints)
        feed_dict.motion_patches = motion_patches
            
        return feed_dict.raw()
    
    def __len__(self):
        return len(self.split_question_ids)

    # get the maximum number of segment in a single sequence across all samples (used for linear temporal projection layer)
    def get_max_num_segments(self):
        max_num_segments = 0
        for _, question in self.questions.items():
            motion_id = question['babel_id']
            if not self.no_gt_segments:
                num_segments = len(self.labels[motion_id])
            else:
                joints = np.load(osp.join(self.joints_root, motion_id, 'joints.npy')) # T, V, C
                total_num_frames = np.shape(joints)[0]
                num_segments = math.ceil(total_num_frames / self.num_frames_per_seg)

            if num_segments > max_num_segments:
                max_num_segments = num_segments
        return max_num_segments
            
class NSTrajDatasetFilterableView(FilterableDatasetView):
    def filter_questions(self, allowed):
        def filt(question):
            return question['query_type'] in allowed
            
        return self.filter(filt, 'filter-question-type[allowed={{{}}}]'.format(','.join(list(allowed))))

    def get_max_num_segments(self):
        return self.owner_dataset.get_max_num_segments()

    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.dataloader import JacDataLoader
        from jactorch.data.collate import VarLengthCollateV2

        collate_guide = {
            'joints': 'concat',
            'answer': 'skip',
            'segment_boundaries': 'skip',
            'filter_boundaries': 'skip',
            'motion_patches': 'skip',
            'program_raw': 'skip',
            'program_seq': 'skip',
            'program_tree': 'skip',
            'program_qsseq': 'skip',
            'program_qstree': 'skip',
        }

        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide))
    
def NSTrajDataset(*args, **kwargs):
    return NSTrajDatasetFilterableView(NSTrajDatasetUnwrapped(*args, **kwargs))



