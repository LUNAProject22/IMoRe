  data_dir=<path_to/BABEL-QA>

  jac-crun 0 nspose/trainval.py \
  --desc nspose/desc_nspose.py \
  --data-dir $data_dir \
  --batch-size 4 \
  --temporal_operator conv1d \
  --no_gt_segments \
  --epochs 100 \
  --featurefusion True \