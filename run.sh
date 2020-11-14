export CUDA_VISIBLE_DEVICES=4

# python src/baseline_main.py --model=mlp --dataset=mnist --gpu=0 --epochs=10
python src/federated_main.py --model cnn \
								--dataset cifar \
								--gpu 0 \
								--iid 1 \
								--epochs 280 \
								--frac 0.1 \
								--local_ep 5 \
								--local_bs 50 \
								--lr 0.15 \
								--decay 0.99 \
								--num_users 100 &&
echo "Congratus! Finished * FILTER* admm training!"




