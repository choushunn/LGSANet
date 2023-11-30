@REM 生成/测试
@REM SGD
python main.py --channel=0 --method=SGD --root_path=./phases
python main.py --channel=1 --method=SGD --root_path=./phases
python main.py --channel=2 --method=SGD --root_path=./phases
@REM DPAC
python main.py --channel=0 --method=DPAC --root_path=./phases
python main.py --channel=1 --method=DPAC --root_path=./phases
python main.py --channel=2 --method=DPAC --root_path=./phases
@REM GS
python main.py --channel=0 --method=GS --root_path=./phases
python main.py --channel=1 --method=GS --root_path=./phases
python main.py --channel=2 --method=GS --root_path=./phases
@REM UNet
python main.py --channel=0 --method=UNet --root_path=./phases --generator_dir=./pretrained_networks
python main.py --channel=1 --method=UNet --root_path=./phases --generator_dir=./pretrained_networks
python main.py --channel=2 --method=UNet --root_path=./phases --generator_dir=./pretrained_networks
@REM HoloNet
python main.py --channel=0 --method=HoloNet --root_path=./phases --generator_dir=./pretrained_networks
python main.py --channel=1 --method=HoloNet --root_path=./phases --generator_dir=./pretrained_networks
python main.py --channel=2 --method=HoloNet --root_path=./phases --generator_dir=./pretrained_networks
@REM Ours
python main.py --channel=0 --method=R2AttUNET --root_path=./phases --generator_dir=./pretrained_networks
python main.py --channel=1 --method=R2AttUNET --root_path=./phases --generator_dir=./pretrained_networks
python main.py --channel=2 --method=R2AttUNET --root_path=./phases --generator_dir=./pretrained_networks

@REM 重建
python eval.py --channel=3 --root_path=./phases/_SGD_ASM --prop_model=ASM
python eval.py --channel=3 --root_path=./phases/_DPAC_ASM --prop_model=ASM
python eval.py --channel=3 --root_path=./phases/_GS_ASM --prop_model=ASM


@REM 训练
python train_holonet.py  --perfect_prop_model=True --run_id=UNet --batch_size=1 --channel=1 --loss_fun=l1
python train_holonet.py  --perfect_prop_model=True --run_id=R2AttU_NET --batch_size=1 --channel=1 --loss_fun=l1
python train_holonet.py  --perfect_prop_model=True --run_id=UNET3P --batch_size=1 --channel=1 --loss_fun=l1