
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

pred_len=24
model_name=NLinear
for t in $(seq 37 39); do
    echo 'Processing '${t}
    target=${t}

    # target='0'
    seq_len=96
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path electricity.csv \
      --model_id Electricity_$seq_len'_'$pred_len \
      --model $model_name \
      --data custom \
      --features S \
      --target $target\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --des 'Exp' \
      --use_gpu false \
      --itr 1 --batch_size 16  --learning_rate 0.001 >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'$pred_len.log 

      seq_len=192
      python -u run_longExp.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path electricity.csv \
        --model_id Electricity_$seq_len'_'$pred_len \
        --model $model_name \
        --data custom \
        --features S \
        --target $target\
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 321 \
        --des 'Exp' \
        --itr 1 --batch_size 16  --learning_rate 0.001 >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'$pred_len.log  

      seq_len=336
      python -u run_longExp.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path electricity.csv \
        --model_id Electricity_$seq_len'_'$pred_len \
        --model $model_name \
        --data custom \
        --features S \
        --target $target\
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 321 \
        --des 'Exp' \
        --itr 1 --batch_size 16  --learning_rate 0.001  >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'$pred_len.log  

      seq_len=720
      python -u run_longExp.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path electricity.csv \
        --model_id Electricity_$seq_len'_'$pred_len \
        --model $model_name \
        --data custom \
        --features S \
        --target $target\
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 321 \
        --des 'Exp' \
        --itr 1 --batch_size 16  --learning_rate 0.001  >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'$pred_len.log  
done