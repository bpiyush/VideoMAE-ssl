  
  
  OUTPUT_DIR='/var/scratch/pbagad/charades_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e800/eval_lr_5e-4_epoch_50'
  DATA_PATH='/var/scratch/pbagad/datasets/Charades/'
  MODEL_PATH="./checkpoints/k400_vitb_ep800_16x5x3.pth"
  
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 \
      --master_port 8010 \
      run_multiclass_finetuning.py \
      --model vit_base_patch16_224 \
      --data_set Charades \
      --nb_classes 157 \
      --data_path ${DATA_PATH} \
      --finetune ${MODEL_PATH} \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size 8 \
      --num_sample 1 \
      --input_size 224 \
      --short_side_size 224 \
      --save_ckpt_freq 10 \
      --num_frames 16 \
      --opt adamw \
      --lr 5e-4 \
      --opt_betas 0.9 0.999 \
      --weight_decay 0.05 \
      --epochs 50 \
      --dist_eval \
      --test_num_segment 2 \
      --test_num_crop 3 \
      --enable_deepspeed 
