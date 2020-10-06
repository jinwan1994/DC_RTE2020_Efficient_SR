# Instruction

## Quickstart (Demo)
```bash

cd codes       # You are now in */Final_version/src
sh demo.sh
```


## Testing
* Runing testing:
```bash
# x2 
python test.py --test_model architecture --LR_path ../datasets/test_data/LR --HR_path ../datasets/test_data/HR --model_path ../pretrained-model/architecture.pth --baseline_path ../pretrained-model/baseline.pth
```

## Training
* Resize the images and convert image files to npy files
```bash
python data/image2npy.py --pathRoot ../datasets/agora_sr_2020.7.28_train/train --pathTo ../datasets/train_data
```
* Run training x2 model
```bash
python train.py --root ../datasets/train_data --scale 2 --loss_type 1*L1 #--pretrained ../pretrained-model/pretrained_model.pth

```