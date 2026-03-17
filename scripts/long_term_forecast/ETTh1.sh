export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_new" ]; then
    mkdir ./logs/LongForecasting_new
fi

seq_len=96
model_name=iTransformer
for pred_len in 96 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id  ETTh1\
    --model  $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7\
    --d_model2 256\
    --d_ff2 256\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/ETTh1_Mlow_$model_name'_96_'$pred_len.log  
done


seq_len=96
model_name=PatchTST
for pred_len in 96 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id  ETTh1\
    --model  $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7\
    --d_model 64\
    --d_ff 128\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/ETTh1_Mlow_$model_name'_96_'$pred_len.log  
done














