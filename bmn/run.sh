export CUDA_VISIBLE_DEVICES=0,1,2,3

start_time=$(date +%s)

python -m paddle.distributed.launch train.py -d

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "4 card static training time is $(($cost_time/60))min $(($cost_time%60))s"
