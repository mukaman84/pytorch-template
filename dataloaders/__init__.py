from dataloaders.datasets import cityscapes, combine_dbs
from dataloaders.datasets_lane import lane_mobis, lane_combine_dbs, Inference_data
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):
    if args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
    elif args.dataset == 'LaneMobis':
        train_set = lane_mobis.LaneMobis(args, split='train')
        val_set = lane_mobis.LaneMobis(args, split='val')
        test_set = lane_mobis.LaneMobis(args, split='test')
        num_class = val_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, val_loader, test_loader, num_class
    elif args.dataset == 'inference':

        val_set = Inference_data.LaneMobis(args, split='val')

        num_class = val_set.NUM_CLASSES

        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return val_loader, num_class
    else:
        raise NotImplementedError

