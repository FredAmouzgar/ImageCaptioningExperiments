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
```
