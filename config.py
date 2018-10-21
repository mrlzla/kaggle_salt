class config:
    kfolds=10
    batch_size=4
    crop_size=(101, 101)
    h=101
    w=101
    output_stride=32
    use_depths=False
    pad=True
    lr=1e-4
    num_epochs=50
    start_epochs=250
    weights_path="weights"
    frozen_epochs=6
    single_fold=True
    num_workers=2
    use_depth=False
    start_epoch=0
    start_fold=0
    start_cycle=0
    num_cycles=2
    use_tta=False
    lr_mult = 0.7
    start_lr=1e-4
    use_resize=True

