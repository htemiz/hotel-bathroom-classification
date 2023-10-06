# hotel-bathroom-classification

Repository for supporting files and outcomes for my paper entitled
[Automatic and Accurate Classification of Hotel Bathrooms from Images with Deep Learning](https://dergipark.org.tr/en/download/article-file/2823031)
published in 
[International Journal of Engineering Research and Development](https://dergipark.org.tr/en/pub/umagd) journal.


Please cite the paper as follows:

*Temiz, Hakan. "Automatic and Accurate Classification of Hotel Bathrooms from Images with Deep Learning.
" International Journal of Engineering Research and Development 14.3 (2022): 211-218.* [https://doi.org/10.29137/umagd.1217004](https://doi.org/10.29137/umagd.1217004)

&nbsp;

### Overview



&nbsp;

### Dataset

The images were downloaded from [TripAdvisor](https://www.tripadvisor.com). In total, 11,116 images were manually
labelled as `good` or `bad`. 10,561 images were separated for training, and 555 images for testing.

7181 bathroom images were classified as good and 3935 images as bad. Of the good and bad images,
6822 and 3739 were reserved for the training, and 359 and 196 were reserved for testing, respectively.

This dataset is shared with the community with the name [HotelBath](https://zenodo.org/record/7340428) (~435MB) in zenodo. 

Details of the dataset:

||Good|Bad|**Total**|
|--|--|--|--|
|Training|6822|3739|10561|
|Test|359|196|555|
|**Total**|7181|3935|11116|

Some sample images that labelled as `good`
![](images/good.jpg)


Some sample images that labelled as `bad`
![](images/bad.jpg)



&nbsp;

### Algorithm

![](images/model.jpg)




### Training



### Evaluation

**True Positive (TP):** Values that are actually positive and predicted positive.

**False Positive (FP):** Values that are actually negative but predicted to be positive.

**False Negative (FN):** Values that are actually positive but predicted to be negative.

**True Negative (TN):** Values that are actually negative and predicted to be negative.

**Accuracy:** The ratio of all correctly predicted positives and negatives.
$$\frac{TP + TN}{TP + TN + FP + FN}$$


&nbsp;

Please feel free to contact me at [htemiz@artvin.edu.tr](mailto:htemiz@artvin.edu.tr) for any further information.


