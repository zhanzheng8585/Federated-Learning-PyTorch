export CUDA_VISIBLE_DEVICES=7

# python src/baseline_main.py --model=mlp --dataset=mnist --gpu=0 --epochs=10
python src/federated_main.py --model=cnn --dataset=cifar --gpu --gpu_id=0 --iid=1 --epochs=50
