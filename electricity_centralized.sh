
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

pred_len=24
model_name=Linear
seq_len=192
    # python -u run_longExp.py \
    #   --is_training 1 \
    #   --root_path ./dataset/ \
    #   --data_path electricity.csv \
    #   --model_id Electricity_$seq_len'_'$pred_len \
    #   --model $model_name \
    #   --data custom \
    #   --features S \
    #   --target $target\
    #   --seq_len $seq_len \
    #   --pred_len $pred_len \
    #   --enc_in 321 \
    #   --des 'Exp' \
    #   --use_gpu false \
    #   --itr 1 --batch_size 16  --learning_rate 0.001 >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'$pred_len.log 

    # seq_len=192
python -u run_longExp_centralized.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 50 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.001 >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'$pred_len.log  

# seq_len=336
# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path electricity.csv \
#   --model_id Electricity_$seq_len'_'$pred_len \
#   --model $model_name \
#   --data custom \
#   --features S \
#   --target $target\
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --enc_in 321 \
#   --des 'Exp' \
#   --itr 1 --batch_size 16  --learning_rate 0.001  >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'$pred_len.log  

#     seq_len=720
#     python -u run_longExp.py \
#       --is_training 1 \
#       --root_path ./dataset/ \
#       --data_path electricity.csv \
#       --model_id Electricity_$seq_len'_'$pred_len \
#       --model $model_name \
#       --data custom \
#       --features S \
#       --target $target\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 321 \
#       --des 'Exp' \
#       --itr 1 --batch_size 16  --learning_rate 0.001  >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'$pred_len.log  
# done