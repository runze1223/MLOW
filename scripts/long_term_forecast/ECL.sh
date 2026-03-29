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
    --data_path electricity.csv \
    --model_id  ECL\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/ECL_mlow_$model_name'_96_'$pred_len.log  
done



seq_len=96
model_name=PatchTST
for pred_len in 96 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id  ECL\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --criterion 'MSE'\
    --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/ECL_mlow_$model_name'_96_'$pred_len.log  
done


seq_len=96
model_name=CycleNet
for pred_len in 96 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id  ECL\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321\
    --d_model 512\
    --cycle 168\
    --des 'Exp' \
    --train_epochs 60\
    --patience 6\
    --lradj 'TST'\
    --criterion 'MAE'\
    --itr 1 --batch_size 64 --learning_rate 0.0005 >logs/LongForecasting_new/ECL_mlow_$model_name'_96_'$pred_len.log  
done



seq_len=96
model_name=NLinear
for pred_len in 96 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id  ECL\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321\
    --des 'Exp' \
    --train_epochs 60\
    --patience 8\
    --lradj 'ST'\
    --criterion 'MAE'\
    --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/ECL_mlow_$model_name'_96_'$pred_len.log  
done
