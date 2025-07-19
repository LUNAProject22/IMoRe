# HuMMan-QA Dataset

To evaluate the generalizability of motion understanding models beyond the small-scale BABEL-QA dataset, we introduce the **HuMMan-QA dataset**, built on top of the [HuMMan-MoGen dataset](https://arxiv.org/pdf/2312.15004).

## Dataset Overview

- **Source**: HuMMan-MoGen contains **6,264 SMPL motion sequences** covering **160 action categories**, with **112,112** detailed temporal and spatial text descriptions.
- **Generation Strategy**: We adopt the motion QA generation methodology proposed in [NSPose](https://github.com/markendo/HumanMotionQA).
- **Challenge**: HuMMan-MoGen lacks segment-level labels for **actions**, **directions**, and **body parts**.
- **Solution**: We leveraged **GPT-4o** with in-context learning to infer these labels, followed by **manual refinement and verification** against the actual motion sequences by the authors.

## Statistics

| Split      | #Questions |
|------------|------------|
| Train      | 2,066      |                   
| Validation | 524        |                   
| Test       | 533        |                   

- **Classes**:
  - **Actions**: 158
  - **Body Parts**: 18
  - **Directions**: 6

> All classes are derived from a total of **1,311 motion sequences**.

## Evaluation Metric

- **Metric**: Accuracy is used as the primary evaluation criterion.

## License
HuMManQA dataset is under [S-Lab License v1.0](https://caizhongang.com/projects/HuMMan/license.txt).


## Download

You can access the HuMMan-QA dataset [here](https://drive.google.com/drive/folders/1JDQyXLfKdDA5o0cvD14IbXKtQFAh1Fad?usp=drive_link)


## Citation
If you use this dataset for your research work, please cite our paper:
```bibtext
To be added... 
```
