export CUDA_VISIBLE_DEVICES=7
export FLAGS_fraction_of_gpu_memory_to_use=0.1

model="mobilenet_v2"
epoch=10
num_workers=3
batch_size=32
train_batchs=100     # -1 means use all samples
test_batchs=-1
lr=0.0001

python quant_dygraph.py \
    --arch=${model} \
    --epoch=${epoch} \
    --num_workers=${num_workers} \
    --train_batchs=${train_batchs} \
    --test_batchs=${test_batchs} \
    --batch_size=${batch_size} \
    --lr=${lr} \
    --enable_quant
