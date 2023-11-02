export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model WDSR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channels 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model WDSR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channels 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model WDSR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channels 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model WDSR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channels 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

