set -ex


# --------------- TDGAN -------------- #
python train.py --dataroot ./db/brain --name Brain_task1 --model TDGAN --netG resnet_9blocks  \
   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 --lambda_reminding_L1 0 --lambda_reminding_perceptual 0  \
   --dataset_mode brain_split --pool_size 0 --gpu_ids 4 --batch_size 8 --num_threads 0 --niter=40 --niter_decay=40 \
   --norm instance --display_freq 256 --print_freq 256 --save_epoch_freq 20 --no_dropout \
   --task_num 1 

python train.py --dataroot ./db/brain --name Brain_TDGAN_task2 --model TDGAN --netG resnet_9blocks  \
   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 --lambda_reminding_L1 100 --lambda_reminding_perceptual 10 \
   --dataset_mode brain_split --pool_size 0 --gpu_ids 4 --batch_size 8 --num_threads 0 --niter=40 --niter_decay=40 \
   --norm instance --display_freq 256 --print_freq 256 --save_epoch_freq 20 --no_dropout \
   --task_num 2 --prev_model_path ./checkpoints/Brain_task1 --prev_model_epoch 80
   
python train.py --dataroot ./db/brain --name Brain_TDGAN_task3 --model TDGAN --netG resnet_9blocks  \
   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 --lambda_reminding_L1 100 --lambda_reminding_perceptual 10 \
   --dataset_mode brain_split --pool_size 0 --gpu_ids 4 --batch_size 8 --num_threads 0 --niter=40 --niter_decay=40 \
   --norm instance --display_freq 256 --print_freq 256 --save_epoch_freq 20 --no_dropout \
   --task_num 3 --prev_model_path ./checkpoints/Brain_TDGAN_task2 --prev_model_epoch 80  
# --------------------------------------------- #


## --------------- Sequential Finetuning --------------- #
#python train.py --dataroot ./db/brain --name Brain_Finetune_task2 --model pix2pix --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 \
#   --dataset_mode brain_split --pool_size 0 --gpu_ids 4  --batch_size 8 --num_threads 0 --niter=40 --niter_decay=40 \
#   --norm instance --display_freq 256 --print_freq 256 --save_epoch_freq 20 --no_dropout \
#   --task_num 2 --prev_model_path ./checkpoints/Brain_task1 --prev_model_epoch 80
#
#python train.py --dataroot ./db/brain --name Brain_Finetune_task3 --model pix2pix --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 \
#   --dataset_mode brain_split --pool_size 0 --gpu_ids 4  --batch_size 8 --num_threads 0 --niter=40 --niter_decay=40 \
#   --norm instance --display_freq 256 --print_freq 256 --save_epoch_freq 20 --no_dropout \
#   --task_num 3 --prev_model_path ./checkpoints/Brain_Finetune_task2 --prev_model_epoch 80
## ---------------------------- #
#
#
## --------------- Joint Learning --------------- #
#python train.py --dataroot ./db/brain --name Brain_joint_task2 --model pix2pix --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 \
#   --dataset_mode brain_joint --pool_size 0 --gpu_ids 4 --batch_size 8 --num_threads 0 --niter=40 --niter_decay=40 \
#   --norm instance --display_freq 256 --print_freq 256 --save_epoch_freq 20 --no_dropout \
#   --task_num 2
#
#python train.py --dataroot ./db/brain --name Brain_joint_task3 --model pix2pix --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 \
#   --dataset_mode brain_joint --pool_size 0 --gpu_ids 4 --batch_size 8 --num_threads 0 --niter=40 --niter_decay=40 \
#   --norm instance --display_freq 256 --print_freq 256 --save_epoch_freq 20 --no_dropout \
#   --task_num 3
## --------------------------------------- #
#
#
## --------------- Local GAN --------------- #
#python train.py --dataroot ./db/brain --name Brain_LocalGAN_BratsT2 --model pix2pix --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 \
#   --dataset_mode brain_split --pool_size 0 --gpu_ids 4 --batch_size 8 --num_threads 0 --niter=40 --niter_decay=40 \
#   --norm instance --display_freq 256 --print_freq 256 --save_epoch_freq 20 --no_dropout \
#   --task_num 2
#
#python train.py --dataroot ./db/brain --name Brain_LocalGAN_BratsT1 --model pix2pix --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 \
#   --dataset_mode brain_split --pool_size 0 --gpu_ids 4 --batch_size 8 --num_threads 0 --niter=40 --niter_decay=40 \
#   --norm instance --display_freq 256 --print_freq 256 --save_epoch_freq 20 --no_dropout \
#   --task_num 3
## --------------------------------------- #