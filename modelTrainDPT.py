import numpy as np
import cv2
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import albumentations as A
from tqdm import tqdm
import segmentation_models_pytorch as smp
from monai.losses import DiceCELoss
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.segmentation import DiceScore, MeanIoU

with open('./dataset3k.json', 'r') as file:
    data = json.load(file)

train_data_pre = data['train_image']
train_gt_pre = data['train_gt']

val_data_pre = data['val_image']
val_gt_pre = data['val_gt']

test_data_pre = data['test_image']
test_gt_pre = data['test_gt']

train_data = [os.path.join('./Dataset', curPath[1:]) for curPath in train_data_pre]
train_gt = [os.path.join('./Dataset', curPath[1:]) for curPath in train_gt_pre]

val_data = [os.path.join('./Dataset', curPath[1:]) for curPath in val_data_pre]
val_gt = [os.path.join('./Dataset', curPath[1:]) for curPath in val_gt_pre]

test_data = [os.path.join('./Dataset', curPath[1:]) for curPath in test_data_pre]
test_gt = [os.path.join('./Dataset', curPath[1:]) for curPath in test_gt_pre]

# Following albumentations documentation for this class: https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/
class CocoCityDataset(Dataset):
    def __init__(self, image, seg, transform):
        self.image = image
        self.seg = seg
        self.transform = transform
    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):
        image = self.image[index]
        seg = self.seg[index]

        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(seg, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            transformedObjects = self.transform(image=img, mask=mask)
            img = transformedObjects['image']
            mask = transformedObjects['mask']

        return img, mask
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device used:', device)

model_name = 'DPT'
img_width, img_height = 512, 512

training_transform = A.Compose([
    A.Resize(img_width, img_height),
    A.OneOf([
        A.HorizontalFlip(),
        A.RandomRotate90(),
    ], p=0.8),
    A.GaussNoise(std_range=(0.1, 0.2), p=0.6),
    A.RandomBrightnessContrast(p=0.6),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    A.ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(img_width, img_height),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    A.ToTensorV2()
])

iterations = [
    {'batch_size': 16, 'learning_rate': 0.00005, 'weight_decay': 1e-3, 'encoder': 'tu-vit_base_patch16_384.augreg_in1k'},
    {'batch_size': 8, 'learning_rate': 1.5936014763508956e-05, 'weight_decay': 0.0002458267379107886, 'encoder': 'tu-vit_base_patch16_384.augreg_in1k'},
    {'batch_size': 32, 'learning_rate': 2.1773329947668437e-05, 'weight_decay': 0.00031862741370895277, 'encoder': 'tu-vit_base_patch16_384.augreg_in1k'},
    {'batch_size': 16, 'learning_rate': 2.2009997712364476e-05, 'weight_decay': 1.1978655611887825e-05, 'encoder': 'tu-vit_base_patch16_384.augreg_in1k'},
    {'batch_size': 8, 'learning_rate': 1.066431080297391e-05, 'weight_decay': 1.843432932323988e-05, 'encoder': 'tu-vit_base_patch16_384.augreg_in1k'},
    {'batch_size': 8, 'learning_rate': 1.3824821274285075e-05, 'weight_decay': 1.1074317287667608e-05, 'encoder': 'tu-vit_base_patch16_384.augreg_in1k'},
    {'batch_size': 8, 'learning_rate': 2.0479928299146545e-05, 'weight_decay': 1.8572095008502867e-05, 'encoder': 'tu-vit_base_patch16_384.augreg_in1k'},
    {'batch_size': 8, 'learning_rate': 4.953460354951554e-05, 'weight_decay': 0.00011349254345547834, 'encoder': 'tu-vit_base_patch16_384.augreg_in1k'},
    {'batch_size': 8, 'learning_rate': 4.696120905701285e-05, 'weight_decay': 0.0005332238221150414, 'encoder': 'tu-vit_base_patch16_384.augreg_in1k'},
    {'batch_size': 16, 'learning_rate': 8.389041139896157e-05, 'weight_decay': 1.0625173891488136e-05, 'encoder': 'tu-vit_base_patch16_384.augreg_in1k'}
]

ablations = [
    {'batch_size': 8, 'learning_rate': 1.3824821274285075e-05, 'weight_decay': 1.1074317287667608e-05, 'encoder': 'tu-vit_small_patch16_384.augreg_in1k'},
    {'batch_size': 8, 'learning_rate': 1.3824821274285075e-05, 'weight_decay': 1.1074317287667608e-05, 'encoder': 'tu-vit_base_patch16_384.augreg_in21k_ft_in1k'},
    {'batch_size': 8, 'learning_rate': 1.3824821274285075e-05, 'weight_decay': 1.1074317287667608e-05, 'encoder': 'tu-vit_large_patch16_384.augreg_in21k_ft_in1k'}
]

for iteration, hyper in enumerate(iterations):
    print(hyper)
    BATCH_SIZE = hyper['batch_size']
    lr = hyper['learning_rate']
    wd = hyper['weight_decay']
    backbone = hyper['encoder']
    
    print("Refreshed Model DPT", backbone)
    
    model = smp.DPT(
        encoder_name=backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=6,
        dynamic_img_size=True
    ).to(device)
    
    training_dataset = CocoCityDataset(train_data, train_gt, training_transform)
    training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    val_dataset = CocoCityDataset(val_data, val_gt, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    test_dataset = CocoCityDataset(test_data, test_gt, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    optimiser = torch.optim.AdamW(model.parameters(), lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.4, patience=4)

    # normalised_weigths = torch.tensor([0.0074, 0.1408, 0.1499, 0.3044, 0.2484, 0.1491])
    # normalised_weigths = normalised_weigths.to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False) #, weight=normalised_weigths 

    EPOCHS = 40

    iteration_loss = []
    training_loss = []
    val_loss = []
    dice_iou = []

    ITERATION = iteration + 7
    checkpoint_path = os.path.join('./CheckpointDPT', 'Model_large21')
    os.makedirs(checkpoint_path, exist_ok=True)
    scaler = GradScaler()

    final_results = None
    final_gt = None

    best_epoch = 0

    lowest_metric_loss = -float('inf')
    early_stop_threshold = 0.001
    epoch_counter = 0
    epoch_patience = 6

    best_result_store = []
    best_gt_store = []

    best_train_result_store = []
    best_train_gt_store = []

    for epoch in range(EPOCHS):
        model.train()
        print(f'Training {epoch+1}/{EPOCHS} Epoch for', model_name)
        train_result_store = []
        train_gt_store = []

        epoch_loss = 0
        batch_loss = []
        for batch in tqdm(training_loader):
            image, mask = batch
            image = image.to(device)
            mask = mask.to(device)

            mask = mask.unsqueeze(1)

            optimiser.zero_grad()
            with autocast():
                results = model(image)
                loss = loss_function(results, mask.long())

            batch_loss.append(loss.item())
            epoch_loss+=loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

        iteration_loss.append(batch_loss)
        training_loss.append(epoch_loss)

        model.eval()
        val_epoch_loss = 0
        result_store = []
        gt_store = []
        with torch.no_grad():
            for i, batch_val in enumerate(tqdm(val_loader)):
                image_val, mask_val = batch_val
                image_val = image_val.to(device)
                mask_val = mask_val.to(device)

                mask_val = mask_val.unsqueeze(1)

                with autocast():
                    results_val = model(image_val)
                    loss_val = loss_function(results_val, mask_val.long())
                val_epoch_loss+=loss_val.item()

                result_store.append(torch.argmax(results_val, dim=1).detach().cpu().long())
                gt_store.append(mask_val.squeeze(1).detach().cpu().long())

        mean_val_loss = val_epoch_loss/len(val_loader)

        final_results_val = torch.cat(result_store, dim=0)
        final_gt_val = torch.cat(gt_store, dim=0)

        dice_score = DiceScore(num_classes=6, average='none', include_background=False, input_format='index', aggregation_level='global')
        calculated_dice = dice_score(final_results_val, final_gt_val)
        print('Validation Dice Score:', calculated_dice)

        iou_score = MeanIoU(num_classes=6, include_background=False, per_class=True, input_format='index')
        calculated_iou = iou_score(final_results_val, final_gt_val)
        print('Validation IOU Score:', calculated_iou)

        mean_metric = 0.5*torch.mean(calculated_dice) + 0.5*torch.mean(calculated_iou)
        print('Valiation combined score:', mean_metric)

        if mean_metric > lowest_metric_loss - early_stop_threshold:
            lowest_metric_loss = mean_metric
            best_epoch = epoch+1
            epoch_counter = 0
            best_result_store = result_store
            best_gt_store = gt_store
        else:
            epoch_counter+=1

        if epoch_counter >= epoch_patience:
            print(f"Early stop at {epoch+1}/{EPOCHS} with training loss: {epoch_loss/len(training_loader)}, and validation loss {val_epoch_loss/len(val_loader)}, combined metric {mean_metric} and early stop epoch {best_epoch}")
            val_loss.append(val_epoch_loss)
            dice_iou.append([calculated_dice, calculated_iou])
            torch.save(model.state_dict(), os.path.join(checkpoint_path, model_name + f'_large21_{epoch+1}.pth'))
            break

        scheduler.step(val_epoch_loss/len(val_loader))
        val_loss.append(val_epoch_loss)
        dice_iou.append([calculated_dice, calculated_iou])
        torch.save(model.state_dict(), os.path.join(checkpoint_path, model_name + f'_large21_{epoch+1}.pth'))

        print(f'Completed epoch {epoch+1}/{EPOCHS} with training loss: {epoch_loss/len(training_loader)}, and validation loss {val_epoch_loss/len(val_loader)}, combined metric {mean_metric} and early stop epoch {best_epoch}')

    final_results = torch.cat(best_result_store, dim=0)
    final_gt = torch.cat(best_gt_store, dim=0)

    dice_score = DiceScore(num_classes=6, average='none', include_background=False, input_format='index', aggregation_level='global')
    calculated_dice = dice_score(final_results, final_gt)
    print('Validation Dice Score:', calculated_dice)

    iou_score = MeanIoU(num_classes=6, include_background=False, per_class=True, input_format='index')
    calculated_iou = iou_score(final_results, final_gt)
    print('Validation IOU Score:', calculated_iou)

    mean_metric = 0.5*torch.mean(calculated_dice) + 0.5*torch.mean(calculated_iou)
    print('Valiation combined score:', mean_metric)

    with open(os.path.join(checkpoint_path, model_name + f'_large21.pkl'), 'wb') as file:
        pickle.dump([iteration_loss, training_loss, val_loss, dice_iou], file)