set -ex


# ---------------- TDGAN -------------------- #
python test_brain.py --dataroot ./db/brain --name Brain_task1 --model pix2pix --netG resnet_9blocks \
    --direction AtoB --dataset_mode nuclei --epoch 80 \
    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 1

# different tasks corresponds to different datasets, thus tested separatedly
python test_brain.py --dataroot ./db/brain --name Brain_TDGAN_task2 --model pix2pix --netG resnet_9blocks \
    --direction AtoB --dataset_mode nuclei --epoch 80 \
    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 1
python test_brain.py --dataroot ./db/brain --name Brain_TDGAN_task2 --model pix2pix --netG resnet_9blocks \
    --direction AtoB --dataset_mode brats --epoch 80 \
    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 2

python test_brain.py --dataroot ./db/brain --name Brain_TDGAN_task3 --model pix2pix --netG resnet_9blocks \
    --direction AtoB --dataset_mode nuclei --epoch 80 \
    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 1
python test_brain.py --dataroot ./db/brain --name Brain_TDGAN_task3 --model pix2pix --netG resnet_9blocks \
    --direction AtoB --dataset_mode brats --epoch 80 \
    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 2
python test_brain.py --dataroot ./db/brain --name Brain_TDGAN_task3 --model pix2pix --netG resnet_9blocks \
    --direction AtoB --dataset_mode bratsT1 --epoch 80 \
    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 3
# --------------------------------------------- #


## -------------- Sequential Finetuning ------------------ #
#python test_brain.py --dataroot ./db/brain --name Brain_Finetune_task2 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei --epoch 80 \
#    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 1
#python test_brain.py --dataroot ./db/brain --name Brain_Finetune_task2 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode brats --epoch 80 \
#    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 2
#
#python test_brain.py --dataroot ./db/brain --name Brain_Finetune_task3 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei --epoch 80  \
#    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 1
#python test_brain.py --dataroot ./db/brain --name Brain_Finetune_task3 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode brats --epoch 80  \
#    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 2
#python test_brain.py --dataroot ./db/brain --name Brain_Finetune_task3 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode bratsT1 --epoch 80 \
#    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 3
## ----------------------------------------------------- #
#
#
## --------------- Joint Learning ------------ #
#python test_brain.py --dataroot ./db/brain  --name Brain_joint_task2 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei --epoch 80 \
#    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 1
#python test_brain.py --dataroot ./db/brain  --name Brain_joint_task2 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode brats --epoch 80 \
#    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 2
#
#python test_brain.py --dataroot ./db/brain  --name Brain_joint_task3 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode nuclei --epoch 80 \
#    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 1
#python test_brain.py --dataroot ./db/brain  --name Brain_joint_task3 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode brats --epoch 80 \
#    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 2
#python test_brain.py --dataroot ./db/brain --name Brain_joint_task3 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode bratsT1 --epoch 80 \
#    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 3
## ------------------------------ #
#
#
## --------------- Local GAN ------------ #
#python test_brain.py --dataroot ./db/brain --name Brain_LocalGAN_BratsT2 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode brats --epoch 80 \
#    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 2
#
#python test_brain.py --dataroot ./db/brain --name Brain_LocalGAN_BratsT1 --model pix2pix --netG resnet_9blocks \
#    --direction AtoB --dataset_mode bratsT1 --epoch 80 \
#    --results_dir results/one_per_task_brain --norm instance --no_dropout --task_num 3
## ----------------------------------------- #
