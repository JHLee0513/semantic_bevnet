def add_common_arguments(parser):
    parser.add_argument('--model_config', type=str, help='path to the model config file.')
    parser.add_argument('--dataset_config', type=str, help='path to the dataset config file.')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_variant', type=str, default='default')
    parser.add_argument('--output', type=str, required=True, help='out directory name.')
    parser.add_argument('--train_device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default='', help='path to the model to resume from.')
    parser.add_argument('--resume_epoch', type=int, default=-1,
        help='epoch to resume from when --resume flag provided.')
    parser.add_argument('-n', '--num_workers', type=int, default=0, help='Dataloader num_workers.')
    parser.add_argument('--log_interval',
        type=int, default=1, help='Log every this number of iterations.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--dataset_type', type=str, default='costmap_4',
        help='Dataset type (mainly for visualization purposes). Could be "costmap_4", "kitti_19", or "heatmap"')
    parser.add_argument('--include_unknown', action='store_true', default=False,
                        help='Include the unknown class.')

    parser.add_argument('--buffer_scans', type=int, default=1,
                        help='How many scans to merge.')
    parser.add_argument('--buffer_scan_stride', type=int, default=1,
                        help='Stride between adjacent scans.')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr_decay_epoch', type=int, default=1,
                        help='Decay learning rate every this number of epochs.')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Learning rate decay.')
    parser.add_argument('--test', action='store_true', help='Run testing.')
    return parser
