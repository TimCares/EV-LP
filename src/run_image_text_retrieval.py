"""
This module is used to perform the image-text retrieval on the test set of the COCO and Flickr30K datasets.
It is used to produce benchmarks as reported in papers BEiT-3, VLMo, FLAVA, CLIP, and others.
"""
import logging
from rich.progress import track
from typing import *
import torch
import os
from pytorch_lightning import LightningModule
from registries import MODEL_REGISTRY, DATAMODULE_REGISTRY
from omegaconf import DictConfig, open_dict
import hydra
import json
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def compute_average_median_rank(images:torch.Tensor, texts:torch.Tensor, iids:torch.Tensor, tiids:torch.Tensor) -> Tuple[float, float]:
    """Computes the average median rank for image-text retrieval, based on the similarity scores between image and text embeddings.
    In total, 25 retrievals are performed, as the images are split into 5 chunks, and the average median rank is computed for each chunk.
    In each chunk, there are 1k images and 5k captions. The 1k images are used 5 times, every time each image is paried with one of its 5 captions.
    This yields a 1k subchunk of images, and a 1k subchunk of captions. In this pairing,
    each image has exactly one correct caption, and vice versa. The rest of the samples are incorrect pairings. Retrieval is then performed
    between the 1k subchunk of images and the 1k subchunk of captions, and the rank of the correct pairing is computed.
    Then the median of the ranks is taken.
    For each chunk, this is repeated 5 times, until each image has been paired with each of its 5 captions once.
    In the end, we compute the average over the 25 median ranks, which is the average median rank.

    This ensures that we have a balanced evaluation, and aims at returning a score comparable to the average median rank in the paper
    "See, Hear, and Read: Deep Aligned Representations" (https://people.csail.mit.edu/yusuf/see-hear-read/). They do not publish
    which exact samples they use for their 1k retrieval, so this method is a reasonable approximation.

    Args:
        images (torch.Tensor): The image embeddings, of shape (num_images, emb_dim).
        texts (torch.Tensor): The text embeddings, of shape (num_texts, emb_dim).
        iids (torch.Tensor): The ids of the images, of shape (num_images,).
        tiids (torch.Tensor): The ids of the texts, of shape (num_texts,).

    Returns:
        Tuple[float, float]: A tuple containing the average median rank for text retrieval (tuple[0]) and image retrieval (tuple[1]).
    """    
    mask = iids.unsqueeze(1) == tiids.unsqueeze(0) # (num_images, num_texts), 1 if image and text are matching (i.e. have the same id), 0 otherwise    
    masks = mask.chunk(5)
    image_chunks = images.chunk(5) # [1k, 1k, 1k, 1k, 1k]
    
    amrs_ir = []
    amrs_tr = []
    for i in range(5): # for each chunk of 1k images
        avg_median_rank_tr, avg_median_rank_ir = amr_for_chunk(image_chunks[i], texts, masks[i])
        amrs_tr.append(avg_median_rank_tr)
        amrs_ir.append(avg_median_rank_ir)
    return sum(amrs_tr) / len(amrs_tr), sum(amrs_ir) / len(amrs_ir)

def amr_for_chunk(images:torch.Tensor, texts:torch.Tensor, mask:torch.Tensor) -> Tuple[float, float]:
    """_summary_

    Args:
        images (torch.Tensor): The image embeddings of the chunk, of shape (1k, emb_dim).
        texts (torch.Tensor): The text embeddings, of shape (25k, emb_dim).
        mask (torch.Tensor): The mask indicating which images and texts are matching, of shape (1k, 5k).

    Returns:
        Tuple[float, float]: The average median rank for text retrieval (tuple[0]) and image retrieval (tuple[1]) on the chunk.
    """    
    selected_indices = []
    # for each image (row), get the indices of the matching texts
    # [:5] -> some images have >5 captions, but in COCO each image is supposed to have 5 captions, so we only take the first 5
    matching_candidates_indices = torch.stack([torch.nonzero(row).squeeze()[:5] for row in mask])

    median_ranks_ir = []
    median_ranks_tr = []
    for i in range(matching_candidates_indices.shape[1]): # will be "range(5)"
        selected_indices = matching_candidates_indices[:, i] # get the i-th matching text for each image
        selected_texts = texts[selected_indices] # get the embeddings of the i-th matching text for each image, shape (1k, emb_dim)

        scores = images @ selected_texts.T # (1k, 1k)
        # get the median rank of the correct pairings (tr), "add(1)" because ranks start at 1, quantile(0.5) is the median
        median_rank_tr = scores.argsort(dim=1, descending=True).argsort(dim=1).diagonal().add(1).float().quantile(0.5).item()
        median_ranks_tr.append(median_rank_tr)
        # get the median rank of the correct pairings (ir), "add(1)" because ranks start at 1, quantile(0.5) is the median
        median_rank_ir = scores.argsort(dim=0, descending=True).argsort(dim=0).diagonal().add(1).float().quantile(0.5).item()
        median_ranks_ir.append(median_rank_ir)

    # median_ranks_tr and median_ranks_ir are lists of length 5, containing the median ranks for each of the 5 "rounds"
    # in each round, each image is paired with one of its 5 captions
    avg_median_rank_tr = sum(median_ranks_tr) / len(median_ranks_tr)
    avg_median_rank_ir = sum(median_ranks_ir) / len(median_ranks_ir)
    return avg_median_rank_tr, avg_median_rank_ir

# following stems mostly from the BEiT3 repo: https://github.com/microsoft/unilm/blob/master/beit3/engine_for_finetuning.py
def compute_scores(
    img_embeds:List[torch.Tensor],
    text_embeds:List[torch.Tensor],
    img_ids:List[torch.Tensor],
    compute_amr:bool=False
) -> Dict[str, float]:
    """Computes R@1, R@5, R@10, and average score for image-text retrieval based on the similarity scores between image and text embeddings.

    Args:
        img_embeds (torch.Tensor): The image embeddings. Still in list form, where each element is one batch of image embeddings.
        text_embeds (torch.Tensor): The text embeddings. Still in list form, where each element is one batch of text embeddings.
        img_ids (torch.Tensor): The ids of the images. Still in list form, where each element is one batch of ids.
            This is important, as a single image can have multiple captions,
            and having one of the correct captions in the top-k results should be considered a correct retrieval.
        compute_amr (bool, optional): Whether to also compute the average median rank of the retrieval, based on
            the paper "See, Hear, and Read: Deep Aligned Representations" (https://people.csail.mit.edu/yusuf/see-hear-read/).
            Only supposed to be used for the COCO test set. Defaults to False.

    Returns:
        Dict[str, float]: The R@1, R@5, R@10 score for image retrieval (tr) and text retrieval (ir), as well as the average score.
            If compute_amr is True, also includes the average median rank for text retrieval (tr_amr) and image retrieval (ir_amr).
    """    
    image_feats = {} # collect all unique image features, and create mapping based on id
    for feats, ids in zip(img_embeds, img_ids):
        for i, _idx in enumerate(ids):
            idx = _idx.item()
            if idx not in image_feats:
                image_feats[idx] = feats[i]

    tiids = torch.cat(img_ids, dim=0) # id of each text/caption
    iids = [] # id of each unique image
    sorted_tensors = []
    # generate sorted batch of image features, based on the ids
    for key in sorted(image_feats.keys()):
        sorted_tensors.append(image_feats[key].view(1, -1))
        iids.append(key)

    img_embeds = torch.cat(sorted_tensors, dim=0) # (num_unique_images, emb_dim)
    text_embeds = torch.cat(text_embeds, dim=0) # (num_texts, emb_dim)

    scores = img_embeds @ text_embeds.t() # (num_unique_images, num_texts)
    iids = torch.LongTensor(iids).to(scores.device)

    # get the indices of the top-k retrieved captions for each image
    topk10 = scores.topk(10, dim=1).indices
    topk5 = scores.topk(5, dim=1).indices
    topk1 = scores.topk(1, dim=1).indices
    
    # get the id of the top-k retrieved captions for each image
    topk10_iids = tiids[topk10]
    topk5_iids = tiids[topk5]
    topk1_iids = tiids[topk1]

    # calculate the retrieval scores for text retrieval (tr)
    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    # get the indices of the top-k retrieved images for each caption
    topk10 = scores.topk(10, dim=0).indices
    topk5 = scores.topk(5, dim=0).indices
    topk1 = scores.topk(1, dim=0).indices

    # get the id of the top-k retrieved images for each caption
    topk10_iids = iids[topk10]
    topk5_iids = iids[topk5]
    topk1_iids = iids[topk1]

    # calculate the retrieval scores for image retrieval (ir)
    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    eval_result = {
        "tr_r10": tr_r10.item() * 100.0, 
        "tr_r5": tr_r5.item() * 100.0, 
        "tr_r1": tr_r1.item() * 100.0, 
        "ir_r10": ir_r10.item() * 100.0, 
        "ir_r5": ir_r5.item() * 100.0, 
        "ir_r1": ir_r1.item() * 100.0, 
        "average_score": 100.0 * (tr_r1 + tr_r5 + tr_r10 + ir_r1 + ir_r5 + ir_r10).item() / 6.0,
    }

    if compute_amr:
        tr_amr, ir_amr = compute_average_median_rank(img_embeds, text_embeds, iids, tiids)
        eval_result['tr_amr'] = tr_amr
        eval_result['ir_amr'] = ir_amr

    logger.info(f'* Eval result = {json.dumps(eval_result)}')
    return eval_result


@torch.no_grad()
def zero_shot_retrieval(model:nn.Module, dataloader, device:str='cuda', compute_amr:bool=False) -> Dict[str, float]:
    """Perform (zero-shot) image-text retrieval on the given dataloader using the given model.

    Args:
        model (nn.Module): The model to use for the image and text embeddings.
        dataloader (_type_): The dataloader to use for the retrieval. Generates batches of image-text pairs.
        device (str, optional): The device to use for the computation. Defaults to 'cuda'.
        compute_amr (bool, optional): Whether to also compute the average median rank. Only supposed to be used
            for the COCO test set. Defaults to False.

    Returns:
        Dict[str, float]: A dictionary containing the retrieval scores for image retrieval (tr) and text retrieval (ir),
            as well as the average score. If compute_amr is True, also includes the average median rank for text retrieval (tr_amr)
            and image retrieval (ir_amr).
    """    
    img_embeds = []
    text_embeds = []
    img_ids = []

    for batch in track(dataloader):
        image = batch['image'].to(device)
        text = batch['text'].to(device)
        padding_mask = batch['padding_mask'].to(device) if 'padding_mask' in batch else None
        # encoding also normalizes the output
        img_emb = model.encode_image(image=image)['x']
        text_emb = model.encode_text(text=text, padding_mask=padding_mask)['x']
        img_embeds.append(img_emb)
        text_embeds.append(text_emb)
        img_ids.append(batch['id'].to(device))

    return compute_scores(img_embeds=img_embeds, text_embeds=text_embeds, img_ids=img_ids, compute_amr=compute_amr)


@hydra.main(version_base=None, config_path=os.path.join("..", "configs"), config_name='coco_flickr_retrieval')
def main(cfg: DictConfig) -> None:
    """Perform (zero-shot) image-text retrieval on the test set of the specified datasets. Usually COCO and Flickr30K.

    Args:
        cfg (DictConfig): The configuration object containing information about the model location, the datasets, etc.
    """    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    datamodules = []

    common_args = cfg.data.common

    for datamodule_key in cfg.data.datamodules.keys(): # COCO and Flickr30K
        dataset_args = cfg.data.datamodules[datamodule_key]
        with open_dict(dataset_args):
            dataset_args.update(common_args)
        dm = DATAMODULE_REGISTRY[datamodule_key](**dataset_args)
        datamodules.append((datamodule_key, dm))
    
    model_cls:LightningModule = MODEL_REGISTRY[cfg.model_name]
    model = model_cls.load_from_checkpoint(cfg.model_path).model
    model = model.to(device)
    # set model to inference mode
    model.requires_grad_(False)
    model.eval()

    for name, dm in datamodules:
        dm.prepare_data()
        dm.setup('test')
        logger.info(f"Zero-shot retrieval on: {name}")
        zero_shot_retrieval(model, dm.test_dataloader(), device, compute_amr=name=='coco_captions')

if __name__ == "__main__":
    main()