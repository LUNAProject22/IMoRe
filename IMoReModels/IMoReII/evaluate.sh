  data_dir=<path_to/BABEL-QA>
  load_path=<path_to/checkpoints/best.pth>

  jac-crun 0 nspose/trainval.py \
  --desc nspose/desc_nspose.py \
  --data-dir $data_dir \
  --batch-size 4 \
  --temporal_operator conv1d \
  --no_gt_segments \
  --load $load_path \
  --evaluate \
  --featurefusion True \