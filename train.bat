@echo off

echo 正在执行第一行代码...
python train_holonet.py --perfect_prop_model=True --run_id=dcUnet --net_name=dcUnet --batch_size=1 --channel=0

echo 第一行代码执行完毕，等待一段时间...
timeout /t 5

echo 正在执行第二行代码...
python train_holonet.py --perfect_prop_model=True --run_id=dcUnet --net_name=dcUnet --batch_size=1 --channel=1

echo 第二行代码执行完毕，等待一段时间...
timeout /t 5

echo 正在执行第三行代码...
python train_holonet.py --perfect_prop_model=True --run_id=dcUnet --net_name=dcUnet --batch_size=1 --channel=2

echo 第三行代码执行完毕，等待一段时间...
timeout /t 5

echo 所有代码执行完毕。