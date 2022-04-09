

# Catastrophic forgetting problem in semi-supervised Semantic segmentation
This repository contains the code for our paper ：Catastrophic forgetting problem in semi-supervised semantic segmentation.
![训练框图](https://user-images.githubusercontent.com/68488554/162558679-0eec28c0-5a28-4f1b-989e-1982f2388c9c.jpg)

Setup
You'll need a CUDA 10, Python3 environment (best on Linux) with PyTorch 1.2.0, TorchVision 0.4.0 and Apex to run the code in this repo.

1. Setup the exact version of Apex & PyTorch & TorchVision for mixed precision training：

```js
pip install https://download.pytorch.org/whl/cu100/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl && pip install https://download.pytorch.org/whl/cu100/torchvision-0.4.0-cp36-cp36m-manylinux1_x86_64.whl  
git clone https://github.com/NVIDIA/apex
cd apex  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

2.  Download the PASCAL VOC 2012 dataset and the Cityscapes dataset. Then prepare the Cityscapes dataset：
```js
python cityscapes_data_list.py
```
Then generate 3 different random splits:

```js
python generate_splits.py
```
Afterwards, your data directory structure should look like these:

```js
├── your_voc_base_dir/VOCtrainval_11-May-2012/VOCdevkit/VOC2012                    
    ├── Annotations 
    ├── ImageSets
    │   ├── Segmentation
    │   │   ├── 1_labeled_0.txt
    │   │   ├── 1_labeled_1.txt
    │   │   └── ... 
    │   └── ... 
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── ...

├── your_city_base_dir                     
    ├── data_lists
    │   ├── 1_labeled_0.txt
    │   ├── 1_labeled_1.txt
    │   └── ...  
    ├── gtFine
    └── leftImage8bit
```
3. Prepare pre-trained weights. For segmentation experiments, we need COCO pre-trained weights same as previous works:

```js
./prepare_coco.sh
```
4. Run the code.For example, run DMT with different pre-trained weights:
```js
./dmt_ple-voc-20-1.sh
```
Run DMT+PLE with difference maximized sampling:

```js
python dms_sample.py
./dmt_ple-voc-106-1lr.sh
```

