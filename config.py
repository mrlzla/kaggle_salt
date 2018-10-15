class config:
    kfolds=10
    batch_size=5
    crop_size=(101, 101)
    h=101
    w=101
    output_stride=32
    use_depths=False
    pad=True
    lr=1e-4
    num_epochs=70
    start_epochs=100
    weights_path="weights"
    frozen_epochs=6
    single_fold=True
    num_workers=2
    use_depth=False
    start_fold=0
    start_cycle=0
    num_cycles=2
    use_tta=False
    lr_mult = 0.4
    start_lr=1e-4#1.6e-5
    use_resize=True

