export CUDA_VISIBLE_DEVICES="4"
python predict.py \
    --attention True \
    --src_lang en --tar_lang vi \
    --num_layers 2 \
    --hidden_size 512 \
    --src_vocab_size 17191 \
    --tar_vocab_size 7709 \
    --batch_size 128 \
    --dropout 0.2 \
    --init_scale  0.1 \
    --max_grad_norm 5.0 \
    --vocab_prefix data/en-vi/vocab \
    --infer_file data/en-vi/tst2013.en \
    --reload_model attention_models/10 \
    --infer_output_file infer_output.txt \
    --beam_size 10 \
    --use_gpu True \
    --eager_run True
