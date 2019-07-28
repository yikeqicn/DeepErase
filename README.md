# DeepErase
- DeepErase is a U-net-like tensorflow sementic segmenation model removing artifacts (lines, boxes, spurious words) from text images extracted from documents. The cleansing of the artifacts enhances OCR performance over the image extractions.

<p align="center">
  <img width="460" height="300" src="https://github.com/yikeqicn/DeepErase/blob/master/example.JPG">
</p>

### Authors
- [Ronny Huang](mailto:wronnyhuang@gmail.com), [Yike Qi](yike.qi.cn@gmail.com) 

### Abstract
- We present a [method](https://github.com/yikeqicn/DeepErase/tree/master/src/DataFactory) to programmatically generate artiﬁcial text images with realistic-looking artifacts, and use them to train the U-net-like model in a totally unsupervised manner.
- The U-net-like model was trained in two modes:
  - [Standalone training](https://github.com/yikeqicn/DeepErase/tree/master/src/CleaningStandaloneModel): Optimize at Unet Segmentation loss only.
  - [Joint training with downstream Recognition model](https://github.com/yikeqicn/DeepErase/tree/master/src/CleaningRecognitionJointModel): Optimize at Unet Segmentation loss + recognition CTC loss. To balance image cleaning and recognition performance. ** [HTR CTC model](https://github.com/wronnyhuang/htr) was used as recognition model.
- Both validation pixel level segmentation accuracies were above 95%.
- Downstream recognition performances were evaluated on validation images and IRS extractions. The IRS extractions were extracted from [NIST sd02 tax forms](https://www.nist.gov/srd/nist-special-database-2), and were not used in model training. The word recognition accuracy were improved and beat the naive Hough cv2 cleaning method.
<p align="center">
  <img  src="https://github.com/yikeqicn/DeepErase/blob/master/Segmentation_Accuracy.JPG">
</p>
<p align="center">
  <img  src="https://github.com/yikeqicn/DeepErase/blob/master/Recognition_Validation_Set.JPG">
</p>
<p align="center">
  <img  src="https://github.com/yikeqicn/DeepErase/blob/master/Recognition_IRS_Set.JPG">
</p>
