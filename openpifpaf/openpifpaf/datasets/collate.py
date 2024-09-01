import torch


def collate_images_anns_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    anns = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    depth_information = torch.utils.data.dataloader.default_collate([b[3] for b in batch])

    return images, anns, metas,depth_information


def collate_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    # depth_gt= torch.utils.data.dataloader.default_collate([b[1][2] for b in batch])
    depth_information = torch.utils.data.dataloader.default_collate([b[3] for b in batch])
    metas = [b[2] for b in batch]
    # targets=[targets,depth_gt]
    # if len(batch[0])==4:
    #     depth_information=[b[3] for b in batch]
    # else:
    #     depth_information=None
    return images, targets, metas, depth_information
