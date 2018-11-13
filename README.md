# Al_surface_defect_detection
This includes my code for Alibaba Tianchi competition:  [Al surface defect detection](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.704833afdEFFgH&raceId=231682).

The competition is aimed at using computer vision techniques to help workers check whether their Al surface products have any defects such as spots, scratches and so on.

### Season1 rank: 96/2972     
##### What I found very useful:  
- InceptionV4(pytorch)
- combine vote(similar to bagging)
- good iteration steps
##### Just so so:  
- data augmentation(horizontal flip)
- Ensemble(Xception, Resnet50, InveptionV3)
##### Decrease my test acc
- Gaussian noise
- Random Rotation(0~8, the larger angle, the worse acc))

### Season2 rank：10/2972     
##### What I found very useful:   
- FasterRcnn&FPN(detectron)
- Larger resize size(960 for maskrcnn, 800 for fasterrcnn)
- bbox vote

