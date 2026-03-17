export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ShortForecasting_new" ]; then
    mkdir ./logs/ShortForecasting_new
fi


seq_len=96
model_name=iTransformer

for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id  PEMS08\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170\
    --des 'Exp' \
    --train_epochs 150\
    --patience 20\
    --itr 1 --batch_size 64 --learning_rate 0.0004 >logs/ShortForecasting_new/PEMS08_mlow2_$model_name'_10_336_96_'$pred_len.log  
done



seq_len=96
model_name=iTransformer

for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id  PEMS03\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358\
    --des 'Exp' \
    --train_epochs 150\
    --patience 20\
    --itr 1 --batch_size 64 --learning_rate 0.0004 >logs/ShortForecasting_new/PEMS03_mlow2_$model_name'_10_336_96_'$pred_len.log  
done



seq_len=96
model_name=iTransformer

for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id  PEMS07\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883\
    --des 'Exp' \
    --train_epochs 150\
    --patience 20\
    --itr 1 --batch_size 64 --learning_rate 0.0004 >logs/ShortForecasting_new/PEMS07_mlow_$model_name'_10_336_96_'$pred_len.log  
done

seq_len=96
model_name=iTransformer
for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id  PEMS04\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307\
    --des 'Exp' \
    --train_epochs 150\
    --patience 20\
    --itr 1 --batch_size 64 --learning_rate 0.0004 >logs/ShortForecasting_new/PEMS04_mlow_$model_name'_10_336_96_'$pred_len.log  
done





seq_len=96
model_name=PatchTST

for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id  PEMS08\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170\
    --des 'Exp' \
    --train_epochs 150\
    --patience 20\
    --itr 1 --batch_size 64 --learning_rate 0.0004 >logs/ShortForecasting_new/PEMS08_mlow_$model_name'_10_336_96_'$pred_len.log  
done



seq_len=96
model_name=PatchTST

for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id  PEMS03\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358\
    --des 'Exp' \
    --train_epochs 150\
    --patience 20\
    --itr 1 --batch_size 64 --learning_rate 0.0004 >logs/ShortForecasting_new/PEMS03_mlow_$model_name'_10_336_96_'$pred_len.log  
done



seq_len=96
model_name=PatchTST

for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id  PEMS07\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883\
    --des 'Exp' \
    --train_epochs 150\
    --patience 20\
    --itr 1 --batch_size 64 --learning_rate 0.0004 >logs/ShortForecasting_new/PEMS07_mlow_$model_name'_10_336_96_'$pred_len.log  
done

seq_len=96
model_name=PatchTST

for pred_len in 12 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1\
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id  PEMS04\
    --model  $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307\
    --des 'Exp' \
    --train_epochs 150\
    --patience 20\
    --itr 1 --batch_size 64 --learning_rate 0.0004 >logs/ShortForecasting_new/PEMS04_mlow_$model_name'_10_336_96_'$pred_len.log  
done