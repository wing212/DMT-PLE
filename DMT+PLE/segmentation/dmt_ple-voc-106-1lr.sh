#!/bin/bash

# lr: left-right, i.e. 2 different samplings of the labeled subset
train_set="106"
sid="1"
exp_name="dmt-voc-106-1lr"
ep="4"
lr="0.001"
c1="5"
c2="5"
i1="5"
i2="5"
phases=("1" "2" "3" "4" "5")
rates=("0.2" "0.4" "0.6" "0.8" "1")
old=("0" "0" "1" "2" "3" "4")
label_pseudo_size=("7" "7" "7" "7" "7")
label_size=("1" "1" "1" "1" "1")
rate_for_Increment=("0.15" "0.15" "0.15" "0.15" "0.15")

python main.py --exp-name=${exp_name}__p0--l --val-num-steps=100 --state=2 --epochs=300 --train-set=${train_set}-l --sets-id=${sid} --continue-from=voc_coco_resnet101.pt --coco --mixed-precision --batch-size-labeled=8 --batch-size-pseudo=0 --seed=1
python main.py --exp-name=${exp_name}__p0--r --val-num-steps=100 --state=2 --epochs=300 --train-set=${train_set}-r --sets-id=${sid} --continue-from=voc_coco_resnet101.pt --coco --mixed-precision --batch-size-labeled=8 --batch-size-pseudo=0 --seed=2

echo dmtphases 1 -- rate 0.2

echo labeling
python main.py --labeling --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--l.pt --coco --mixed-precision --batch-size-labeled=8 --label-ratio=${rates[$i]}

echo training
python main.py --exp-name=${exp_name}__p${phases[$i]}--r --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--r.pt --coco --mixed-precision --epochs=${ep} --gamma1=${c1} --gamma2=${c2} --lr=${lr} --batch-size-labeled=1 --batch-size-pseudo=7 --seed=1
  
echo labeling
python main.py --labeling --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--r.pt --coco --mixed-precision --batch-size-labeled=8 --label-ratio=${rates[$i]}

echo training
python main.py --exp-name=${exp_name}__p${phases[$i]}--l --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--l.pt --coco --mixed-precision --epochs=${ep} --gamma1=${i1} --gamma2=${i2} --lr=${lr} --batch-size-labeled=1 --batch-size-pseudo=7 --seed=2
        

for ((i=1;i<=4;i++)); do
  echo ${phases[$i]}--${rates[$i]}

  echo labeling
  python main_PLE.py --labeling  --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--l.pt --oldfilename=${exp_name}__p${old[$i]}--l.pt --coco --mixed-precision --batch-size-labeled=8 --label-ratio=${rates[$i]} --Increment=${rate_for_Increment[$i]}

  echo training
  python main_PLE.py --exp-name=${exp_name}__p${phases[$i]}--r --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--r.pt --coco --mixed-precision --epochs=${ep} --gamma1=${c1} --gamma2=${c2} --lr=${lr} --batch-size-labeled=${label_size[$i]} --batch-size-pseudo=${label_pseudo_size[$i]} --seed=1

  echo labeling
  python main_PLE.py --labeling --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--r.pt --oldfilename=${exp_name}__p${old[$i]}--r.pt --coco --mixed-precision --batch-size-labeled=8 --label-ratio=${rates[$i]} --Increment=${rate_for_Increment[$i]}

  echo training
  python main_PLE.py --exp-name=${exp_name}__p${phases[$i]}--l --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--l.pt --coco --mixed-precision  --epochs=${ep} --gamma1=${i1} --gamma2=${i2} --lr=${lr} --batch-size-labeled=${label_size[$i]} --batch-size-pseudo=${label_pseudo_size[$i]} --seed=2
done
