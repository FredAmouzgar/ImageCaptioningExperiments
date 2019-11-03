# ImageCaptioningExperiments

This code implements our image captioning experiments which compares the original SCST method and our proposed clipped-SC and interpolated ones inspired by the PPO algorithm.

Our code is implemented on top of the unofficial PyTorch implementation of SCST. Special thanks go to the developer, <a href="https://github.com/ruotianluo">Ruotian(RT) Luo</a>, who helped us in our implementation.

How to run the code:
* The SCST code works only on python 2.7. Thus, our code follows the same standard.
* Clone the SCST code and its requirements such as MS COCO dataset and trained ResNet model.
```console
# Cloning the SCST
git clone https://github.com/ruotianluo/self-critical.pytorch.git
# Cloning our code
git clone https://github.com/FredAmouzgar/ImageCaptioningExperiments.git
# Replace our files with the original SCST code
mv ImageCaptioningExperiments/* self-critical.pytorch/*
cd self-critical.pytorch
# Pre-training for 30 epochs with Cross-Entropy using FC model
python train.py --id fc --caption_model fc --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_fc --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30
# Clipped-SC training for 60 epochs (30 epochs after pre-training) using FC model
python2 train.py --id fc_rl --caption_model fc --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-5 --start_from log_fc_rl --checkpoint_path log_fc_rl --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --self_critical_after 30 --max_epochs 60 --sc_lambda 0.75 --clipped_lambda 0.25
```
<hr>
Credit: <a href="https://arxiv.org/abs/1707.06347">PPO paper</a>, <a href="https://arxiv.org/abs/1612.00563">SCST paper</a>, <a href="https://github.com/ruotianluo/self-critical.pytorch">SCST implementation</a>
