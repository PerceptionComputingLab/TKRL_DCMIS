cd /Share8/zhuzhanshi/tkrl/
source activate CL
clear

export PYTHONPATH=/Share8/zhuzhanshi/tkrl/mp:$PYTHONPATH
# hyperparamer search mm

# probes search with different multiply batch size

python main.py   --approach pcd --epochs 30 --experiment-name  polyp-pcd02   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-pcd.log --multiply-probes 2
python main.py   --approach pcd --epochs 30 --experiment-name  prostate-pcd02   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-pcd.log --multiply-probes 2
python main.py   --approach pcd --epochs 30 --experiment-name  hippocampus-pcd02   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-pcd.log --multiply-probes 2

python main.py   --approach pcd --epochs 30 --experiment-name  polyp-pcd03   --batch-size 16 --device-ids 1 --dataset polyp --resume-from polyp-seq > log/polyp-pcd.log --multiply-probes 3
python main.py   --approach pcd --epochs 30 --experiment-name  prostate-pcd03   --batch-size 16 --device-ids 1 --dataset prostate --resume-from prostate-seq > log/prostate-pcd.log --multiply-probes 3
python main.py   --approach pcd --epochs 30 --experiment-name  hippocampus-pcd03   --batch-size 16 --device-ids 1 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-pcd.log --multiply-probes 3



# hyperparamer search on vma at higher masking ratio,such as 0.75 and 0.50
python main.py   --approach rmae --epochs 30 --experiment-name  polyp-rmae50   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-rmae.log --mask-ratio 0.5
python main.py   --approach rmae --epochs 30 --experiment-name  prostate-rmae50   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-rmae.log --mask-ratio 0.5
python main.py   --approach rmae --epochs 30 --experiment-name  hippocampus-rmae50   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-rmae.log --mask-ratio 0.5

python main.py   --approach vma --epochs 30 --experiment-name  polyp-vma50   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-vma.log --mask-ratio 0.5
python main.py   --approach vma --epochs 30 --experiment-name  prostate-vma50   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-vma.log --mask-ratio 0.5
python main.py   --approach vma --epochs 30 --experiment-name  hippocampus-vma50   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-vma.log --mask-ratio 0.5

python main.py   --approach rmae --epochs 30 --experiment-name  polyp-rmae75   --batch-size 16 --device-ids 1 --dataset polyp --resume-from polyp-seq > log/polyp-rmae.log --mask-ratio 0.75
python main.py   --approach rmae --epochs 30 --experiment-name  prostate-rmae75   --batch-size 16 --device-ids 1 --dataset prostate --resume-from prostate-seq > log/prostate-rmae.log --mask-ratio 0.75
python main.py   --approach rmae --epochs 30 --experiment-name  hippocampus-rmae75   --batch-size 16 --device-ids 1 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-rmae.log --mask-ratio 0.75

python main.py   --approach vma --epochs 30 --experiment-name  polyp-vma75   --batch-size 16 --device-ids 1 --dataset polyp --resume-from polyp-seq > log/polyp-vma.log --mask-ratio 0.75
python main.py   --approach vma --epochs 30 --experiment-name  prostate-vma75   --batch-size 16 --device-ids 1 --dataset prostate --resume-from prostate-seq > log/prostate-vma.log --mask-ratio 0.75
python main.py   --approach vma --epochs 30 --experiment-name  hippocampus-vma75   --batch-size 16 --device-ids 1 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-vma.log --mask-ratio 0.75



# random mae

python main.py   --approach rmae --epochs 30 --experiment-name  polyp-rmae   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-rmae.log
python main.py   --approach rmae --epochs 30 --experiment-name  prostate-rmae   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-rmae.log
python main.py   --approach rmae --epochs 30 --experiment-name  hippocampus-rmae   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-rmae.log

python main.py   --approach rmae --epochs 30 --experiment-name  polyp-rmae5   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-rmae.log --mask-ratio 0.05
python main.py   --approach rmae --epochs 30 --experiment-name  prostate-rmae5   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-rmae.log --mask-ratio 0.05
python main.py   --approach rmae --epochs 30 --experiment-name  hippocampus-rmae5   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-rmae.log --mask-ratio 0.05

python main.py   --approach rmae --epochs 30 --experiment-name  polyp-rmae2   --batch-size 16 --device-ids 1 --dataset polyp --resume-from polyp-seq > log/polyp-rmae.log --mask-ratio 0.2
python main.py   --approach rmae --epochs 30 --experiment-name  prostate-rmae2   --batch-size 16 --device-ids 1 --dataset prostate --resume-from prostate-seq > log/prostate-rmae.log --mask-ratio 0.2
python main.py   --approach rmae --epochs 30 --experiment-name  hippocampus-rmae2   --batch-size 16 --device-ids 1 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-rmae.log --mask-ratio 0.2

python main.py   --approach rmae --epochs 30 --experiment-name  polyp-rmae3   --batch-size 16 --device-ids 1 --dataset polyp --resume-from polyp-seq > log/polyp-rmae.log --mask-ratio 0.3
python main.py   --approach rmae --epochs 30 --experiment-name  prostate-rmae3   --batch-size 16 --device-ids 1 --dataset prostate --resume-from prostate-seq > log/prostate-rmae.log --mask-ratio 0.3
python main.py   --approach rmae --epochs 30 --experiment-name  hippocampus-rmae3   --batch-size 16 --device-ids 1 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-rmae.log --mask-ratio 0.3



# hyperparamer search
python main.py   --approach pcd --epochs 30 --experiment-name  polyp-pcd-95   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-pcd.log --boundary 0.95
python main.py   --approach pcd --epochs 30 --experiment-name  prostate-pcd-95   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-pcd.log --boundary 0.95
python main.py   --approach pcd --epochs 30 --experiment-name  hippocampus-pcd-95   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-pcd.log --boundary 0.95

python main.py   --approach pcd --epochs 30 --experiment-name  polyp-pcd-9   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-pcd.log --boundary 0.9
python main.py   --approach pcd --epochs 30 --experiment-name  prostate-pcd-9   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-pcd.log --boundary 0.9
python main.py   --approach pcd --epochs 30 --experiment-name  hippocampus-pcd-9   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-pcd.log --boundary 0.9

python main.py   --approach pcd --epochs 30 --experiment-name  polyp-pcd-8   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-pcd.log --boundary 0.8
python main.py   --approach pcd --epochs 30 --experiment-name  prostate-pcd-8   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-pcd.log --boundary 0.8
python main.py   --approach pcd --epochs 30 --experiment-name  hippocampus-pcd-8   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-pcd.log --boundary 0.8

python main.py   --approach vma --epochs 30 --experiment-name  polyp-vma-5   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-vma.log --mask-ratio 0.05
python main.py   --approach vma --epochs 30 --experiment-name  prostate-vma-5   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-vma.log --mask-ratio 0.05
python main.py   --approach vma --epochs 30 --experiment-name  hippocampus-vma-5   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-vma.log --mask-ratio 0.05

python main.py   --approach vma --epochs 30 --experiment-name  polyp-vma-2   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-vma.log --mask-ratio 0.2
python main.py   --approach vma --epochs 30 --experiment-name  prostate-vma-2   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-vma.log --mask-ratio 0.2
python main.py   --approach vma --epochs 30 --experiment-name  hippocampus-vma-2   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-vma.log --mask-ratio 0.2

python main.py   --approach vma --epochs 30 --experiment-name  polyp-vma-3   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-vma.log --mask-ratio 0.3
python main.py   --approach vma --epochs 30 --experiment-name  prostate-vma-3   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-vma.log --mask-ratio 0.3
python main.py   --approach vma --epochs 30 --experiment-name  hippocampus-vma-3   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-vma.log --mask-ratio 0.3

# tkrl
python main.py   --approach tkrl --epochs 30 --experiment-name  polyp-tkrl   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-tkrl.log
python main.py   --approach tkrl --epochs 30 --experiment-name  prostate-tkrl   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-tkrl.log
python main.py   --approach tkrl --epochs 30 --experiment-name  optici-tkrl   --batch-size 16 --device-ids 0  --dataset optic --target-class i --resume-from optici-seq > log/optici-tkrl.log
python main.py   --approach tkrl --epochs 30 --experiment-name  optico-tkrl   --batch-size 16 --device-ids 0 --dataset optic --target-class o --resume-from optico-seq > log/optico-tkrl.log
python main.py   --approach tkrl --epochs 30 --experiment-name  mmi-tkrl --batch-size 16 --device-ids 1  --dataset mm --target-class i --resume-from mmi-seq > log/mmi-tkrl.log
python main.py   --approach tkrl --epochs 30 --experiment-name  mmo-tkrl --batch-size 16 --device-ids 1  --dataset mm --target-class o --resume-from mmo-seq > log/mmo-tkrl.log
python main.py   --approach tkrl --epochs 30 --experiment-name  mmr-tkrl --batch-size 16 --device-ids 1  --dataset mm --target-class r --resume-from mmr-seq > log/mmr-tkrl.log
python main.py   --approach tkrl --epochs 30 --experiment-name  hippocampus-tkrl   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-tkrl.log

# pcd
python main.py   --approach pcd --epochs 30 --experiment-name  polyp-pcd   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-pcd.log
python main.py   --approach pcd --epochs 30 --experiment-name  prostate-pcd   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-pcd.log
python main.py   --approach pcd --epochs 30 --experiment-name  optici-pcd   --batch-size 16 --device-ids 0  --dataset optic --target-class i --resume-from optici-seq > log/optici-pcd.log
python main.py   --approach pcd --epochs 30 --experiment-name  optico-pcd   --batch-size 16 --device-ids 0 --dataset optic --target-class o --resume-from optico-seq > log/optico-pcd.log
python main.py   --approach pcd --epochs 30 --experiment-name  mmi-pcd --batch-size 16 --device-ids 1  --dataset mm --target-class i --resume-from mmi-seq > log/mmi-p c d.log
python main.py   --approach pcd --epochs 30 --experiment-name  mmo-pcd --batch-size 16 --device-ids 1  --dataset mm --target-class o --resume-from mmo-seq > log/mmo-pcd.log
python main.py   --approach pcd --epochs 30 --experiment-name  mmr-pcd --batch-size 16 --device-ids 1  --dataset mm --target-class r --resume-from mmr-seq > log/mmr-pcd.log
python main.py   --approach pcd --epochs 30 --experiment-name  hippocampus-pcd   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-pcd.log

# vma
python main.py   --approach vma --epochs 30 --experiment-name  polyp-vma   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-vma.log
python main.py   --approach vma --epochs 30 --experiment-name  prostate-vma   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-vma.log
python main.py   --approach vma --epochs 30 --experiment-name  optici-vma   --batch-size 16 --device-ids 0  --dataset optic --target-class i --resume-from optici-seq > log/optici-vma.log
python main.py   --approach vma --epochs 30 --experiment-name  optico-vma   --batch-size 16 --device-ids 0 --dataset optic --target-class o --resume-from optico-seq > log/optico-vma.log
python main.py   --approach vma --epochs 30 --experiment-name  mmi-vma --batch-size 16 --device-ids 0  --dataset mm --target-class i --resume-from mmi-seq > log/mmi-vma.log
python main.py   --approach vma --epochs 30 --experiment-name  mmo-vma --batch-size 16 --device-ids 0  --dataset mm --target-class o --resume-from mmo-seq > log/mmo-vma.log
python main.py   --approach vma --epochs 30 --experiment-name  mmr-vma --batch-size 16 --device-ids 0  --dataset mm --target-class r --resume-from mmr-seq > log/mmr-vma.log
python main.py   --approach vma --epochs 30 --experiment-name  hippocampus-vma   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-vma.log


# polyp
python main.py   --approach seq --epochs 30 --experiment-name  polyp-seq   --batch-size 16 --device-ids 0 --dataset polyp > log/polyp-seq.log
python main.py   --approach joint --epochs 30 --experiment-name  polyp-joint   --batch-size 16 --device-ids 0 --dataset polyp > log/polyp-joint.log 
python main.py   --approach mas --epochs 30 --experiment-name  polyp-mas   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-mas.log 
python main.py   --approach ewc --epochs 30 --experiment-name  polyp-ewc   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-ewc.log 

python main.py   --approach kd --epochs 30 --experiment-name  polyp-kd   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-kd.log
python main.py   --approach mib --epochs 30 --experiment-name  polyp-mib   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-mib.log
python main.py   --approach plop --epochs 30 --experiment-name  polyp-plop   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-plop.log
python main.py   --approach ted --epochs 30 --experiment-name  polyp-ted   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-ted.log

python main.py   --approach pcd --epochs 30 --experiment-name  polyp-pcd   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-pcd.log
python main.py   --approach vma --epochs 30 --experiment-name  polyp-vma   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-vma.log
python main.py   --approach tkrl --epochs 30 --experiment-name  polyp-tkrl   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-tkrl.log


# prostate
python main.py   --approach seq --epochs 30 --experiment-name  prostate-seq   --batch-size 16 --device-ids 0 --dataset prostate > log/prostate-seq.log
python main.py   --approach joint --epochs 30 --experiment-name  prostate-joint   --batch-size 16 --device-ids 0 --dataset prostate > log/prostate-joint.log 
python main.py   --approach mas --epochs 30 --experiment-name  prostate-mas   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-mas.log 
python main.py   --approach ewc --epochs 30 --experiment-name  prostate-ewc   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-ewc.log 

python main.py   --approach kd --epochs 30 --experiment-name  prostate-kd   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-kd.log
python main.py   --approach mib --epochs 30 --experiment-name  prostate-mib   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-mib.log
python main.py   --approach plop --epochs 30 --experiment-name  prostate-plop   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-plop.log
python main.py   --approach ted --epochs 30 --experiment-name  prostate-ted   --batch-size 16 --device-ids 0 --dataset prostate --resume-from prostate-seq > log/prostate-ted.log


# optici
python main.py   --approach seq --epochs 30 --experiment-name  optici-seq   --batch-size 16 --device-ids 0  --dataset optic --target-class i > log/optici-seq.log
python main.py   --approach joint --epochs 30 --experiment-name  optici-joint   --batch-size 16 --device-ids 0 --dataset optic --target-class i > log/optici-joint.log
python main.py   --approach mas --epochs 30 --experiment-name  optici-mas   --batch-size 16 --device-ids 0  --dataset optic --target-class i --resume-from optici-seq > log/optici-mas.log
python main.py   --approach ewc --epochs 30 --experiment-name  optici-ewc   --batch-size 16 --device-ids 0  --dataset optic --target-class i --resume-from optici-seq > log/optici-ewc.log

python main.py   --approach kd --epochs 30 --experiment-name  optici-kd   --batch-size 16 --device-ids 0  --dataset optic --target-class i --resume-from optici-seq > log/optici-kd.log
python main.py   --approach mib --epochs 30 --experiment-name  optici-mib   --batch-size 16 --device-ids 0  --dataset optic --target-class i --resume-from optici-seq > log/optici-mib.log
python main.py   --approach plop --epochs 30 --experiment-name  optici-plop   --batch-size 16 --device-ids 0  --dataset optic --target-class i --resume-from optici-seq > log/optici-plop.log
python main.py   --approach ted --epochs 30 --experiment-name  optici-ted   --batch-size 16 --device-ids 0  --dataset optic --target-class i --resume-from optici-seq > log/optici-ted.log

# optico
python main.py   --approach seq --epochs 30 --experiment-name  optico-seq   --batch-size 16 --device-ids 0  --dataset optic --target-class o > log/optico-seq.log
python main.py   --approach joint --epochs 30 --experiment-name  optico-joint   --batch-size 16 --device-ids 0 --dataset optic --target-class o > log/optico-joint.log
python main.py   --approach mas --epochs 30 --experiment-name  optico-mas   --batch-size 16 --device-ids 0 --dataset optic --target-class o --resume-from optico-seq > log/optico-mas.log
python main.py   --approach ewc --epochs 30 --experiment-name  optico-ewc   --batch-size 16 --device-ids 0 --dataset optic --target-class o --resume-from optico-seq > log/optico-ewc.log

python main.py   --approach kd --epochs 30 --experiment-name  optico-kd   --batch-size 16 --device-ids 0 --dataset optic --target-class o --resume-from optico-seq > log/optico-kd.log
python main.py   --approach mib --epochs 30 --experiment-name  optico-mib   --batch-size 16 --device-ids 0 --dataset optic --target-class o --resume-from optico-seq > log/optico-mib.log
python main.py   --approach plop --epochs 30 --experiment-name  optico-plop   --batch-size 16 --device-ids 0 --dataset optic --target-class o --resume-from optico-seq > log/optico-plop.log
python main.py   --approach ted --epochs 30 --experiment-name  optico-ted   --batch-size 16 --device-ids 0 --dataset optic --target-class o --resume-from optico-seq > log/optico-ted.log


# mmi
python main.py   --approach seq --epochs 30 --experiment-name  mmi-seq   --batch-size 16 --device-ids 0 --dataset mm --target-class i > log/mmi-seq.log
python main.py   --approach joint --epochs 30 --experiment-name  mmi-joint   --batch-size 16 --device-ids 0 --dataset mm --target-class i > log/mmi-joint.log
python main.py   --approach mas --epochs 30 --experiment-name  mmi-mas --batch-size 16 --device-ids 0  --dataset mm --target-class i --resume-from mmi-seq > log/mmi-mas.log
python main.py   --approach ewc --epochs 30 --experiment-name  mmi-ewc --batch-size 16 --device-ids 0  --dataset mm --target-class i --resume-from mmi-seq > log/mmi-ewc.log

python main.py   --approach kd --epochs 30 --experiment-name  mmi-kd --batch-size 16 --device-ids 0  --dataset mm --target-class i --resume-from mmi-seq > log/mmi-kd.log
python main.py   --approach mib --epochs 30 --experiment-name  mmi-mib --batch-size 16 --device-ids 0  --dataset mm --target-class i --resume-from mmi-seq > log/mmi-mib.log
python main.py   --approach plop --epochs 30 --experiment-name  mmi-plop --batch-size 16 --device-ids 0  --dataset mm --target-class i --resume-from mmi-seq > log/mmi-plop.log
python main.py   --approach ted --epochs 30 --experiment-name  mmi-ted --batch-size 16 --device-ids 0  --dataset mm --target-class i --resume-from mmi-seq > log/mmi-ted.log

# mmo
python main.py   --approach seq --epochs 30 --experiment-name  mmo-seq   --batch-size 16 --device-ids 0  --dataset mm --target-class o > log/mmo-seq.log
python main.py   --approach joint --epochs 30 --experiment-name  mmo-joint   --batch-size 16 --device-ids 0 --dataset mm --target-class o > log/mmo-joint.log
python main.py   --approach mas --epochs 30 --experiment-name  mmo-mas --batch-size 16 --device-ids 0 --dataset mm --target-class o --resume-from mmo-seq > log/mmo-mas.log
python main.py   --approach ewc --epochs 30 --experiment-name  mmo-ewc --batch-size 16 --device-ids 0 --dataset mm --target-class o --resume-from mmo-seq > log/mmo-ewc.log

python main.py   --approach kd --epochs 30 --experiment-name  mmo-kd --batch-size 16 --device-ids 0 --dataset mm --target-class o --resume-from mmo-seq > log/mmo-kd.log
python main.py   --approach mib --epochs 30 --experiment-name  mmo-mib --batch-size 16 --device-ids 0 --dataset mm --target-class o --resume-from mmo-seq > log/mmo-mib.log
python main.py   --approach plop --epochs 30 --experiment-name  mmo-plop --batch-size 16 --device-ids 0 --dataset mm --target-class o --resume-from mmo-seq > log/mmo-plop.log
python main.py   --approach ted --epochs 30 --experiment-name  mmo-ted --batch-size 16 --device-ids 0 --dataset mm --target-class o --resume-from mmo-seq > log/mmo-ted.log

# mmr
python main.py   --approach seq --epochs 30 --experiment-name  mmr-seq   --batch-size 16 --device-ids 0 --dataset mm --target-class r > log/mmr-seq.log
python main.py   --approach joint --epochs 30 --experiment-name  mmr-joint   --batch-size 16 --device-ids 0 --dataset mm --target-class r > log/mmr-joint.log
python main.py   --approach mas --epochs 30 --experiment-name  mmr-mas   --batch-size 16 --device-ids 0 --dataset mm --target-class r --resume-from mmr-seq > log/mmr-mas.log
python main.py   --approach ewc --epochs 30 --experiment-name  mmr-ewc   --batch-size 16 --device-ids 0 --dataset mm --target-class r --resume-from mmr-seq > log/mmr-ewc.log

python main.py   --approach kd --epochs 30 --experiment-name  mmr-kd   --batch-size 16 --device-ids 0 --dataset mm --target-class r --resume-from mmr-seq > log/mmr-kd.log
python main.py   --approach mib --epochs 30 --experiment-name  mmr-mib   --batch-size 16 --device-ids 0 --dataset mm --target-class r --resume-from mmr-seq > log/mmr-mib.log
python main.py   --approach plop --epochs 30 --experiment-name  mmr-plop   --batch-size 16 --device-ids 0 --dataset mm --target-class r --resume-from mmr-seq > log/mmr-mas.log
python main.py   --approach ted --epochs 30 --experiment-name  mmr-ted   --batch-size 16 --device-ids 0 --dataset mm --target-class r --resume-from mmr-seq > log/mmr-ted.log


# hippocampus
python main.py   --approach seq --epochs 30 --experiment-name  hippocampus-seq   --batch-size 16 --device-ids 0 --dataset hippocampus > log/hippocampus-seq.log
python main.py   --approach joint --epochs 30 --experiment-name  hippocampus-joint   --batch-size 16 --device-ids 0 --dataset hippocampus > log/hippocampus-joint.log
python main.py   --approach mas --epochs 30 --experiment-name  hippocampus-mas   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-mas.log
python main.py   --approach ewc --epochs 30 --experiment-name  hippocampus-ewc   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-ewc.log

python main.py   --approach kd --epochs 30 --experiment-name  hippocampus-kd   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-kd.log
python main.py   --approach mib --epochs 30 --experiment-name  hippocampus-mib   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-mib.log
python main.py   --approach plop --epochs 30 --experiment-name  hippocampus-plop   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-plop.log
python main.py   --approach ted --epochs 30 --experiment-name  hippocampus-ted   --batch-size 16 --device-ids 0 --dataset hippocampus --resume-from hippocampus-seq > log/hippocampus-ted.log
