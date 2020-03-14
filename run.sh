#!/usr/bin/env bash


cd experiments/


data_dir=fewshot_shards/
data_dir=/Users/sean/data/semantic_image_segmentation/FSS-1000/fewshot_shards/
name=meta-eval_GPUHO_EffLab_rsd-stages-3-6_`date +%s`
checkpoint_dir=EfficientLab-6-3_FOMAML-star_checkpoint

python ../run_metasegnet.py --fss_1000 --image_size 224 \
    --pretrained \
    --rsd 2 4 --l2 \
    --foml --foml-tail 5 \
    --final_layer_dropout_rate 0.5 --augment --aug_rate 0.5 \
    --sgd --loss_name bce_dice --inner-batch 8 --learning-rate 0.0005 --train-shots 10 --inner-iters 59 --learning_rate_scheduler fixed \
    --meta-iters 50000 --meta-batch 5 \
    --eval-interval 500 --serially_eval_all_test_tasks --eval-samples 2 --shots 5 --eval-batch 8 --eval-iters 59 --transductive \
    --model_name efficientlab  --sgd --meta-step 0.1 --meta-step-final 0.00001 \
    --checkpoint ${checkpoint_dir} --data-dir ${data_dir}  #  2>&1 | tee log_${name}.txt
