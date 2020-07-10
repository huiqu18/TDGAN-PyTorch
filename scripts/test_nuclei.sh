set -ex


# ---------------- TDGAN -------------------- #
python test_TDGAN_nuclei.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_task1 --model pix2pix --netG resnet_9blocks \
    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
    --results_dir results/one_per_task_nuclei --norm instance --no_dropout --task_num 1

python test_TDGAN_nuclei.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_TDGAN_task2 --model pix2pix --netG resnet_9blocks \
    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
    --results_dir results/one_per_task_nuclei --norm instance --no_dropout --task_num 2

python test_TDGAN_nuclei.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_TDGAN_task3 --model pix2pix --netG resnet_9blocks \
    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
    --results_dir results/one_per_task_nuclei --norm instance --no_dropout --task_num 3

python test_TDGAN_nuclei.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_TDGAN_task4 --model pix2pix --netG resnet_9blocks \
    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
    --results_dir results/one_per_task_nuclei --norm instance --no_dropout --task_num 4
# --------------------------------------------- #


## -------------- Sequential Finetuning ------------------ #
#python test_TDGAN_nuclei.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_Finetune_task2 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
#    --results_dir results/one_per_task_nuclei --norm instance --no_dropout --task_num 2
#
#python test_TDGAN_nuclei.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_Finetune_task3 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
#    --results_dir results/one_per_task_nuclei --norm instance --no_dropout --task_num 3
#
#python test_TDGAN_nuclei.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_Finetune_task4 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
#    --results_dir results/one_per_task_nuclei --norm instance --no_dropout --task_num 4
## ----------------------------------------------------- #
#
#
## --------------- Joint Learning ------------ #
#python test_TDGAN_nuclei.py --dataroot ./db/nuclei/for_gan_training  --name Nuclei_joint_task2 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
#    --results_dir results/one_per_task_nuclei --norm instance --no_dropout --task_num 2
##
#python test_TDGAN_nuclei.py --dataroot ./db/nuclei/for_gan_training  --name Nuclei_joint_task3 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
#    --results_dir results/one_per_task_nuclei --norm instance --no_dropout --task_num 3
#
#python test_TDGAN_nuclei.py --dataroot ./db/nuclei/for_gan_training  --name Nuclei_joint_task4 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
#    --results_dir results/one_per_task_nuclei --norm instance --no_dropout --task_num 4
## ----------------------------------------------------- #
#
#
## --------------- Local GAN ------------ #
#python test_pix2pix_nuclei.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_LocalGAN_breast --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
#    --results_dir results/one_per_task_nuclei --norm instance --no_dropout --task_num 2
##
#python test_pix2pix_nuclei.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_LocalGAN_kidney --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
#    --results_dir results/one_per_task_nuclei --norm instance --no_dropout --task_num 3
#
#python test_pix2pix_nuclei.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_LocalGAN_prostate --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
#    --results_dir results/one_per_task_nuclei --norm instance --no_dropout --task_num 4
## ----------------------------------------------------- #
#
#
## ----- 2 discriminators per task, TDGAN ----- #
#python test_TDGAN_2DperTask_nuclei.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_TDGAN_twoDis_task1 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
#    --results_dir results/two_per_task_nuclei --norm instance --no_dropout --task_num 1
#
#python test_TDGAN_2DperTask_nuclei.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_TDGAN_twoDis_task2 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
#    --results_dir results/two_per_task_nuclei --norm instance --no_dropout --task_num 2
### ----------------------------------------------------- #
#
#
## ----- 2 discriminators per task, Sequential Finetuning ----- #
#python test_TDGAN_2DperTask_nuclei.py --dataroot ./db/nuclei/for_gan_training --name Nuclei_Finetune_twoDis_task2 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei_split --epoch 300 \
#    --results_dir results/two_per_task_nuclei --norm instance --no_dropout --task_num 2
## ---------------------------------------------------------------------
