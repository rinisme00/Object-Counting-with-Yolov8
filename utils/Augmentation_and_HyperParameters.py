# utils/aug_hyp.py
def get_aug_hyp():
    """
    Augmentation & hyperparameters for Ultralytics train().
    Values here match your table's defaults.
    """
    return dict(
        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.40,
        degrees=0.0,
        translate=0.10,
        scale=0.50,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.50,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,         # seg only
        copy_paste_mode="flip", # seg only
        auto_augment="randaugment",  # classify only
        erasing=0.4,                 # classify only
    )
