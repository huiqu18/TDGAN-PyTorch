set -ex


# --------------- TDGAN -------------- #
python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_task1 --model TDGAN --netG resnet_9blocks  \
   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 --lambda_reminding_L1 0 --lambda_reminding_perceptual 0 \
   --dataset_mode nuclei_split --pool_size 0 --gpu_ids 2 --batch_size 8 --num_threads 0 --niter=150 --niter_decay=150 \
   --norm instance --display_freq 64 --print_freq 64 --save_epoch_freq 50 --no_dropout \
   --task_num 1

python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_TDGAN_task2 --model TDGAN --netG resnet_9blocks  \
   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 --lambda_reminding_L1 100 --lambda_reminding_perceptual 10 \
   --dataset_mode nuclei_split --pool_size 0 --gpu_ids 2 --batch_size 8 --num_threads 0 --niter=150 --niter_decay=150 \
   --norm instance --display_freq 64 --print_freq 64 --save_epoch_freq 50 --no_dropout \
   --task_num 2 --prev_model_path ./checkpoints/Nuclei_task1 --prev_model_epoch 300

python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_TDGAN_task3 --model TDGAN --netG resnet_9blocks  \
   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 --lambda_reminding_L1 100 --lambda_reminding_perceptual 10 \
   --dataset_mode nuclei_split --pool_size 0 --gpu_ids 2 --batch_size 4 --num_threads 0 --niter=150 --niter_decay=150 \
   --norm instance --display_freq 64 --print_freq 64 --save_epoch_freq 50 --no_dropout \
   --task_num 3 --prev_model_path ./checkpoints/Nuclei_TDGAN_task2 --prev_model_epoch 300

python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_TDGAN_task4 --model TDGAN --netG resnet_9blocks  \
   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 --lambda_reminding_L1 100 --lambda_reminding_perceptual 10 \
   --dataset_mode nuclei_split --pool_size 0 --gpu_ids 2 --batch_size 4 --num_threads 0 --niter=150 --niter_decay=150 \
   --norm instance --display_freq 64 --print_freq 64 --save_epoch_freq 50 --no_dropout \
   --task_num 4 --prev_model_path ./checkpoints/Nuclei_TDGAN_task3 --prev_model_epoch 300
# --------------------------------------------- #


## --------------- Sequential Finetuning --------------- #
#python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_Finetune_task2 --model pix2pix --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 \
#   --dataset_mode nuclei_split --pool_size 0 --gpu_ids 2 --batch_size 8 --num_threads 0 --niter=150 --niter_decay=150 \
#   --norm instance --display_freq 64 --print_freq 64 --save_epoch_freq 50 --no_dropout \
#   --task_num 2 --prev_model_path ./checkpoints/Nuclei_task1 --prev_model_epoch 300
#
#python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_Finetune_task3 --model pix2pix --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 \
#   --dataset_mode nuclei_split --pool_size 0 --gpu_ids 2 --batch_size 8 --num_threads 0 --niter=150 --niter_decay=150 \
#   --norm instance --display_freq 64 --print_freq 64 --save_epoch_freq 50 --no_dropout \
#   --task_num 3 --prev_model_path ./checkpoints/Nuclei_Finetune_task2 --prev_model_epoch 300 \
#
#python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_Finetune_task4 --model pix2pix --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 \
#   --dataset_mode nuclei_split --pool_size 0 --gpu_ids 2 --batch_size 8 --num_threads 0 --niter=150 --niter_decay=150 \
#   --norm instance --display_freq 64 --print_freq 64 --save_epoch_freq 50 --no_dropout \
#   --task_num 4 --prev_model_path ./checkpoints/Nuclei_Finetune_task3 --prev_model_epoch 300
## ---------------------- #


## --------------- Joint Learning --------------- #
#python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_joint_task2 --model pix2pix --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 \
#   --dataset_mode nuclei_joint --pool_size 0 --gpu_ids 2 --batch_size 8 --num_threads 0 --niter=150 --niter_decay=150 \
#   --norm instance --display_freq 256 --print_freq 256 --save_epoch_freq 50 --no_dropout \
#   --task_num 2
#
#python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_joint_task3 --model pix2pix --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 \
#   --dataset_mode nuclei_joint --pool_size 0 --gpu_ids 2 --batch_size 8 --num_threads 0 --niter=150 --niter_decay=150 \
#   --norm instance --display_freq 256 --print_freq 256 --save_epoch_freq 50 --no_dropout \
#   --task_num 3
#
#python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_joint_task4 --model pix2pix --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 \
#   --dataset_mode nuclei_joint --pool_size 0 --gpu_ids 2 --batch_size 8 --num_threads 0 --niter=150 --niter_decay=150 \
#   --norm instance --display_freq 256 --print_freq 256 --save_epoch_freq 50 --no_dropout \
#   --task_num 4
## --------------------------------------- #
#
#
## --------------- Local GAN --------------- #
#python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_LocalGAN_breast --model pix2pix --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 \
#   --dataset_mode nuclei_split --pool_size 0 --gpu_ids 2 --batch_size 8 --num_threads 0 --niter=150 --niter_decay=150 \
#   --norm instance --display_freq 64 --print_freq 64 --save_epoch_freq 50 --no_dropout \
#   --task_num 2
#
#python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_LocalGAN_kidney --model pix2pix --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 \
#   --dataset_mode nuclei_split --pool_size 0 --gpu_ids 2 --batch_size 8 --num_threads 0 --niter=150 --niter_decay=150 \
#   --norm instance --display_freq 64 --print_freq 64 --save_epoch_freq 50 --no_dropout \
#   --task_num 3
#
#python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_LocalGAN_prostate --model pix2pix --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 \
#   --dataset_mode nuclei_split --pool_size 0 --gpu_ids 2 --batch_size 8 --num_threads 0 --niter=150 --niter_decay=150 \
#   --norm instance --display_freq 64 --print_freq 64 --save_epoch_freq 50 --no_dropout \
#   --task_num 4
## --------------------------------------------- #
#
#
#
## ----- 2 discriminators per task, TDGAN ----- #
#python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_TDGAN_twoDis_task1 --model TDGAN_multiD --netG resnet_9blocks \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 --lambda_reminding_L1 0 --lambda_reminding_perceptual 0 \
#   --dataset_mode nuclei_split --pool_size 0 --gpu_ids 2 --batch_size 4 --num_threads 0 --niter=150 --niter_decay=150 \
#   --norm instance --display_freq 64 --print_freq 64 --save_epoch_freq 50 --no_dropout \
#   --task_num 1
#
#python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_TDGAN_twoDis_task2 --model TDGAN_multiD --netG resnet_9blocks  \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 --lambda_reminding_L1 100 --lambda_reminding_perceptual 10 \
#   --dataset_mode nuclei_split --pool_size 0 --gpu_ids 2 --batch_size 4 --num_threads 0 --niter=150 --niter_decay=150 \
#   --norm instance --display_freq 64 --print_freq 64 --save_epoch_freq 50 --no_dropout \
#   --task_num 2 --prev_model_path ./checkpoints/Nuclei_TDGAN_twoDis_task1 --prev_model_epoch 300
## ------------------------------------------------------- #
#
#
## ----- 2 discriminators per task, Sequential Finetuning ----- #
#python train.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_Finetune_twoDis_task2 --model TDGAN_multiD --netG resnet_9blocks --no_lifelong_training \
#   --direction AtoB --lambda_digesting_L1 100 --lambda_digesting_perceptual 10 --lambda_reminding_L1 100 --lambda_reminding_perceptual 10 \
#   --dataset_mode nuclei_split --pool_size 0 --gpu_ids 2 --batch_size 4 --num_threads 0 --niter=150 --niter_decay=150 \
#   --norm instance --display_freq 64 --print_freq 64 --save_epoch_freq 50 --no_dropout \
#   --task_num 2 --prev_model_path ./checkpoints/Nuclei_TDGAN_twoDis_task1 --prev_model_epoch 300
## ------------------------------------------------------- #
