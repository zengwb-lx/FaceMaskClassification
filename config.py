class DefaultConfigs(object):
    # set default configs, if you don't understand, don't modify
    seed = 666            # set random seed
    workers = 8           # set number of data loading workers (default: 4)
    beta1 = 0.9           # adam parameters beta1
    beta2 = 0.999         # adam parameters beta2
    mom = 0.9             # momentum parameters
    wd = 1e-4             # weight-decay
    resume = None  #'./checkpoints/mobilenet-best_model.pth.tar'         # path to latest checkpoint (default: none),should endswith ".pth" or ".tar" if used
    evaluate = False      # just do evaluate
    start_epoch = 0       # deault start epoch is zero,if use resume change it
    split_online = False  # split dataset to train and val online or offline

    # set changeable configs, you can change one during your experiment
    dataset = "./AI/dataset/"  # dataset folder with train and val
    test_folder = "./AI/dataset/"      # test images' folder
    submit_example = "./AI/submit_example.csv"    # submit example file
    checkpoints = "./checkpoints/"        # path to save checkpoints
    log_dir = "./logs/"                   # path to save log files
    submits = "./submits/"                # path to save submission files
    bs = 64               # batch size
    lr = 2e-3             # learning rate
    epochs = 250           # train epochs
    input_size = 640      # model input size or image resied
    num_classes = 2       # num of classes
    gpu_id = "0"          # default gpu id
    model_name = "resnext50_32x4d"      # model name to use
    optim = "sgd"        # "adam","radam","novograd",sgd","ranger","ralamb","over9000","lookahead","lamb"
    fp16 = True          # use float16 to train the model
    opt_level = "O1"      # if use fp16, "O0" means fp32，"O1" means mixed，"O2" means except BN，"O3" means only fp16
    keep_batchnorm_fp32 = False  # if use fp16,keep BN layer as fp32
    loss_func = "CrossEntropy" # "CrossEntropy"、"FocalLoss"、"LabelSmoothCE"
    lr_scheduler = "on_loss"  # lr scheduler method,"adjust","on_loss","on_acc","step"

    
configs = DefaultConfigs()
