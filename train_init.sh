
# train init Tree
python CDTree.py --experiment test_ant_cifar10 --subexperiment 0313 --dataset cifar10 \
--router_ver 2 --router_ngf 64 --router_k 3 --transformer_ver 5 --transformer_ngf 64 \
--transformer_k 3 --solver_ver 2 --batch_norm --maxdepth 4 --batch-size 256 \
--augmentation_on --scheduler step_lr --criteria avg_valid_loss --epochs_patience 15 \
--epochs_node 30 --epochs_finetune 30 --seed 0 --num_workers 0 --gpu 1 --lr 0.01 \
--testonly true


# grow the cdtree (final version)
python CDTree_backup.py --experiment test_ant_cifar10 --subexperiment 0313 --dataset cifar10 \
--router_ver 2 --router_ngf 64 --router_k 3 --transformer_ver 5 --transformer_ngf 64 \
--transformer_k 3 --solver_ver 2 --batch_norm --maxdepth 6 --batch-size 256 \
--augmentation_on --scheduler step_lr --criteria avg_valid_loss --epochs_patience 15 \
--epochs_node 30 --epochs_finetune 30 --epochs_da 15 --seed 0 --num_workers 0 --gpu 1 --lr 0.01

