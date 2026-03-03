### Image-Segmentation-Model-Comparison
- Comparing Performance of UNET++ and Dense Prediction Transformer (DPT) models on the COCO and Cityscapes dataset.
- This was part of my CE6190 coursework and hence the information below is mostly self-cited from my report.

## Dataset processing

# Dataset selection
- The project uses the Common Objects in Context (COCO) and Cityscapes datasets and the targets objects selected were: People, Cars, Trucks, Buses, and Motorcycles.
- For the training data, 3000 images were taken from COCO and Cityscapes datasets. 375 images were then selected for the validation and test sets each.
- Given that images containing single, large centralised objects are often found in COCO datasets, images selected are required to have at least 2 target classes to prevent such cases.

# Dataset normalisation
- Images passed into the UNET++ model were normalised to a mean of (0.485, 0.456, 0.406) and a std of (0.229, 0.224, 0.225) and scaled to values between 0 and 1.
- Images passed into DPT were normalised to a mean and std of (0.5, 0.5, 0.5).
- All image augmentation and normalisation utilised the albumentations library (Buslaev et al., 2020)
- Albumentations documentation was consulted: https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/

# Results:
- UNET++ (with ResNet-50 encoders) achieved high dice scores for the car, bus and person categories (0.8690, 0.7986, 0.8109) but lower dice scores for the motorcycle and truck catgories (0.6770, 0.5641).
- DPT similarly achived high dice scores for the car, bus, and person categories (0.8575, 0.8042, 0.7785) and lower dice scores for the motorcycle and truck categories (0.7125, 0.6321).
- Both models achieved relatively similar results but the DPT model outperformed the UNET++ model in the motorcycle and truck categories.

# Reference
Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020). Albumentations: Fast and Flexible Image Augmentations. Information, 11(2), 125. https://doi.org/10.3390/info11020125
