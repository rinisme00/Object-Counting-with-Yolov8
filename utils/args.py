def get_train_settings():
    """
    Return a dict of Ultralytics train() kwargs.
    Edit defaults here to suit your project.
    Only widely-supported keys are included; others may be ignored by the task.
    """
    return dict(
        # ---- Loss weights / task knobs
        pose=12.0,          # pose-only (ignored for detect)
        kobj=2.0,           # pose/keypoint objness (ignored for detect)
        dfl=1.5,
        cls=0.5,
        box=7.5,

        # ---- Core training
        epochs=100,
        imgsz=640,
        batch=16,           # int, -1 (auto ~60% VRAM), or 0.xx (fraction)
        data=None,          # set in CLI
        model=None,         # set in CLI (we pass YOLOv8 .pt in train_yolov8_auto.py)
        device=None,        # auto in main
        workers=8,
        patience=100,
        single_cls=False,
        seed=0,
        resume=False,
        fraction=1.0,
        classes=None,       # list of class ids to train (None=all)
        name=None,
        project=None,
        time=None,          # hours limit

        # ---- Optim / LR / Regularization
        cos_lr=False,
        momentum=0.937,
        warmup_epochs=3.0,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
        lr0=0.01,
        lrf=0.01,
        weight_decay=5e-4,
        optimizer="auto",

        # ---- Misc behaviours
        exist_ok=False,
        plots=False,
        save_period=-1,
        freeze=None,        # int or list
        deterministic=True,
        val=True,
        save=True,
        compile=False,      # True/'default'/'reduce-overhead'/...
        profile=False,
        multi_scale=False,
        rect=False,
        cache=False,        # True/'ram'/'disk'/False
        amp=True,
        dropout=0.0,
        mask_ratio=4,       # seg-only
        close_mosaic=10,
        pretrained=True,
        overlap_mask=True,  # seg-only
    )