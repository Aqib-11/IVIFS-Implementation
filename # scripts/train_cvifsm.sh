# scripts/train_cvifsm.sh
python -m ivifs.run_train_cvifsm --dataset FMB --epochs 400 --lr 6e-5 --batch 3
python -m ivifs.run_train_cvifsm --dataset MSRS --epochs 1000 --lr 6e-5 --batch 3

# scripts/train_ifam.sh
python -m ivifs.run_train_ifam --epochs 800 --lr 1e-6 --batch 4
