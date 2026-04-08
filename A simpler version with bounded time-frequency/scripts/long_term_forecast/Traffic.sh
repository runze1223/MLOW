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
    --data_path traffic.csv \
    --model_id  Traffic\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862\
    --des 'Exp' \
    --e_layer 4\
    --train_epochs 150\
    --patience 10\
    --lradj 'TST'\
    --std 1\
    --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/Traffic_Mlow_$model_name'_96_'$pred_len.log  
done

seq_len=96
model_name=PatchTST
for pred_len in 96 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id  Traffic\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862\
    --des 'Exp' \
    --e_layer 4\
    --train_epochs 150\
    --patience 10\
    --lradj 'TST'\
    --std 1\
    --criterion 'MSE'\
    --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/Traffic_Mlow_$model_name'_96_'$pred_len.log  
done

seq_len=96
model_name=CycleNet
for pred_len in 96 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id  Traffic\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862\
    --des 'Exp' \
    --d_model 512\
    --cycle 168\
    --train_epochs 60\
    --patience 6\
    --lradj 'TST'\
    --std 1\
    --itr 1 --batch_size 32 --learning_rate 0.0002 >logs/LongForecasting_new/Traffic_Mlow_$model_name'_96_'$pred_len.log  
done


seq_len=96
model_name=NLinear
for pred_len in 96 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id  Traffic\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862\
    --des 'Exp' \
    --e_layer 5\
    --train_epochs 60\
    --patience 8\
    --lradj 'ST'\
    --std 1\
    --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting_new/Traffic_Mlow_$model_name'_96_'$pred_len.log  
done

