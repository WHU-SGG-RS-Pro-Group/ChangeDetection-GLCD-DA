# GLCD-DA：使用具有多种注意力的全局-局部网络用于光学和SAR图像变化检测
**Jie Li, Meiru Wu, Liupeng Lin, Qiangqiang Yuan, Huanfeng Shen**

**Abstract**－Change detection based on optical and synthetic aperture radar (SAR) images is a common technique for extracting change information of the same geographical area from heterogeneous images of different phases under severe weather conditions. Due to the distinct imaging mechanisms of optical and SAR data, change detection between these modalities presents unique challenges. Existing deep learning-based methods for optical-SAR change detection lack targeted attention to multi-level features and underutilize the complementary spatial and semantic information. To address this, a Global and Local Change Detection network with Diversified Attention (GLCD-DA) is proposed for optical and SAR imagery change detection. Under this framework, an image transfer network is employed to convert optical and SAR images into homogeneous images, which serve as input data for the subsequent change detection network. In the change detection network, an Edge-Injected Fusion Module (EIFM) is designed to inject edge information into low-level features to fully exploit spatial detail. Furthermore, a Global-Local Interaction Module (GLIM) that couples CNN with Transformer is proposed to effectively learn both local and global information from high-level features. Finally, a Feature Fusion Module (FFM) is employed to fuse the difference features at different levels. The training process of the two subnetworks is constrained by the image transfer loss function and the change detection loss function, respectively. Experiments on the Gloucester Ⅰ, California and Gloucester Ⅱ datasets show that the proposed method achieves state-of-the-art performance over comparison algorithms in optical and SAR image change detection, which can significantly reduce omissions and false detections.

# Python Requirements
* torch==2.0.1+cu117
* torchvision==0.15.2+cu117
* einops==0.8.1
* matplotlib==3.8.2
* numpy==2.2.3
# Change Detection Datasets
* GloucesterⅠ：http://www-labs.iro.umontreal.ca/~mignotte/ResearchMaterial/#NACCL
* California：https://sites.google.com/view/luppino/data
* GloucesterⅡ：http://www-labs.iro.umontreal.ca/~mignotte/ResearchMaterial/#FPMSMCD
* **Data structure**<br>
├─A <br>
├─B <br>
├─label<br>
└─list<br>
`A`: images of t1 phase;<br>
`B`:images of t2 phase;<br>
`label`: label maps;<br>
`list`: the image file names in the change detection dataset are recorded in .train.txt, val.txt and test.txt respectively.
# Usage
> ## Training
> To train a network, run:
> ```
> python main_cd.py
> --project_name=California
> --checkpoint_root=CHECKPOINTS_CALIFORNI #The training model is saved in ./CHECKPOINTS_CALIFORNI/California
> --batch_size=8
> --img_size=256
> --net_G=base_transformer_pos_s4  #modelname
> --loss=multi #celoss+diceloss+edgeloss
> ```
> ## Evaluate
> To evaluate a network, run:
> ```
> python eval.py
> ```
> ## Predict
> We have some samples from the California dataset in the folder for prediction `./SAMPLES_CALIFORNIA`<br>
> You can find our pretrained model of California in `./CHECKPOINTS_CALIFORNI/California`<br>
> Get started as follows:
> ```
> python predict.py
> ```
> After that, you can find the prediction results in `./RESULTS/CALIFORNIA` and  `./RESULTS/COLOR_CALIFORNIA`<br>
> For heatmap visualization of change detection results you can find it in  `./MiddleFeature/CD_Map`


