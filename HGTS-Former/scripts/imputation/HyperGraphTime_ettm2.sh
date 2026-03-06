export CUDA_VISIBLE_DEVICES=4
model_name=HGTS-Former
token_num=32
token_len=32
seq_len=$[$token_num*$token_len]
# training one model with a context length
for mask_rate in 0.125 0.25 0.375 0.50
do
python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2 \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --train_epochs 20 \
  --d_model 256 \
  --d_ff 1024 \
  --gpu 0 \
  --cosine \
  --use_norm \
  --e_layers 1 \
  --valid_last \
  --edge_num 24 \
  --mask_rate $mask_rate 
done