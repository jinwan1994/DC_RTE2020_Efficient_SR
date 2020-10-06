# Data processing
python data/image2npy.py --pathRoot ../datasets/agora_sr_2020.7.28_train/train --pathTo ../datasets/train_data
# Train
python train.py --root ../datasets/train_data --scale 2 --loss_type 1*L1 #--pretrained ../pre_train/model.pth
# Test
#python test.py --test_model architecture --LR_path ../datasets/test_data/LR --HR_path ../datasets/test_data/HR --model_path ../pretrained-model/architecture.pth --baseline_path ../pretrained-model/baseline.pth