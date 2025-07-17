# IMoRe Models

## Setup
Run the following commands to clone the project and install necessary dependencies.

```bash
  git clone https://github.com/LUNAProject22/IMoRe.git
  cd IMoRe
  conda create -n imore python=3.8.2
  conda activate imore
  pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

Install [Jacinle](https://github.com/vacancy/Jacinle).
```bash
  git clone https://github.com/vacancy/Jacinle --recursive
  export PATH=<path_to_jacinle>/bin:$PATH
```

## Data
- Due to data distribution policies of AMASS and BABEL, we follow the steps given in [BABEL-QA dataset](https://github.com/markendo/HumanMotionQA/tree/master/BABEL-QA) page, to generate the BABEL-QA dataset.
- We perform additional preprocessing on top of the original BABEL-QA annotations. The resulting preprocessed data is available in the [IPGRM_preprocessing](./IPGRM_preprocessing/IPGRM_formatted_data) folder. 
- Please place the [IPGRM_preprocessing](./IPGRM_preprocessing/IPGRM_formatted_data) directory inside both the IMoReI and IMoReII repositories.
- To replicate the preprocessing pipeline, run the following commands:,
```bash
  python ./IPGRM_preprocessing/convert_BABELQA_to_GQA_format.py
  python ./IPGRM_preprocessing/create_BABELQA_qs_program_pairs.py
  python ./IPGRM_preprocessing/create_inputs_for_MMN_resoning_module.py
  python ./IPGRM_preprocessing/calculate_mean_variance.py
```

## Pre-trained Motion ViT Weights
Download the pre-trained [HumanML3D Motion ViT](https://huggingface.co/line-corporation/MotionPatches/tree/main/HumanML3D) of and place it inside the ./pretrained_models directory of both the IMoReI and IMoReII repositories.

## Evaluation

Weights for our trained IMoRe I and II mdoels can be found [here](https://drive.google.com/drive/folders/1JDQyXLfKdDA5o0cvD14IbXKtQFAh1Fad?usp=drive_link).

To evaluate the models, run,
```bash
bash evaluate.sh
```

## Training
To train the models from scratch, run,
```bash
bash train.sh
```

## Citation
```bibtext
To be added...
```

## Acknowledgments
Our code is inspired by [NSPose](https://github.com/markendo/HumanMotionQA).
We obtained the motion weights from [Motion ViT](https://github.com/line/MotionPatches).
