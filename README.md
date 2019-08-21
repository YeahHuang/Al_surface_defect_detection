# Al_surface_defect_detection
This includes my code for Tianchi competition:  [Al surface defect detection](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.704833afdEFFgH&raceId=231682). (held by Alibaba company)

The competition is aimed at using computer vision techniques to help workers check whether their Al surface products have any defects such as spots, scratches and so on.

### Season1(clasification) rank: 96/2972   
##### What I found very useful:  
- InceptionV4(pytorch)
- combine vote(similar to bagging)
- good iteration steps
##### Just so so:  
- data augmentation(horizontal flip)
- Ensemble(Xception, Resnet50, InveptionV3)
##### Decrease my test acc
- All other augmentation. especially random rotation(0~8, the larger angle, the worse acc))

### Season2(localization) rank：10/2972     
##### What I found very useful:   
- FasterRcnn&FPN(detectron)
- Larger resize size(960 for maskrcnn, 800 for FasterRcnn)
- bbox vote
- Adam instead of SGD
- lower the thresh

##### Just so so:
- Mask-Rcnn(keras tf)
- YoloV3
- FasterRcnn(tf)
- Emsemble(FPN, faster-rcnn, mask-rcnn)
- Soft-nms(since few defects have overlap)
- bbox-vote strategy(ID, AVG, IOU_AVG)
- Delete mini batch(since spots are super small)
- Data augmentation( train&test scales,  flip, small rotation)

##### Decrease my test acc:
- my own bbox vote( similar to softer-nms,  a combination of iou and confidence)
- bbox combination ( similar to [this](https://github.com/mirzaevinom/data_science_bowl_2018/blob/master/codes/predict.py) from kaggle big bowl 2018)
- Use larger size(1920x2560), more data augmentation(5 scales etc.)... 

##### What I didn't have time to try:
- SNIPER
- Cascade-rcnn
- maskrcnn(X152 backbone)

#### Possible useful links:
- [Official discussion forum]( https://tianchi.aliyun.com/forum/?spm=5176.12281976.0.0.555e2881eM6ncv#raceId=231682)
- [DCN github link](https://github.com/msracver/Deformable-ConvNets)
- [1st place's article sharing](https://zhuanlan.zhihu.com/p/50548998)
