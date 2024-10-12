from .unimodal_datamodules import BaseDataModule
from datasets_ import COCOCaptions, Flickr30Dataset, ConceptualCaptions
from registries import register_datamodule

@register_datamodule(name='COCOCaptions')
class COCOCaptionsDataModule(BaseDataModule):
    def __init__(self,
                data_path,
                num_max_bpe_tokens,
                task="captioning",
                color_jitter=None,
                beit_transforms=False,
                crop_scale=(0.6, 1.0),
                text_token_mask_prob=0.0,
                *args,
                **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.task = task
        self.color_jitter = color_jitter
        self.beit_transforms = beit_transforms
        self.crop_scale = crop_scale
        self.text_token_mask_prob = text_token_mask_prob

    def set_train_dataset(self):
        self.train_dataset = COCOCaptions(data_path=self.data_path,
                                          split='train',
                                          num_max_bpe_tokens=self.num_max_bpe_tokens,
                                          task=self.task,
                                          color_jitter=self.color_jitter,
                                          beit_transforms=self.beit_transforms,
                                          crop_scale=self.crop_scale,
                                          text_token_mask_prob=self.text_token_mask_prob,)

    def set_val_dataset(self):
        self.val_dataset = COCOCaptions(data_path=self.data_path,
                                        split='val',
                                        num_max_bpe_tokens=self.num_max_bpe_tokens,
                                        task=self.task,
                                        color_jitter=False,
                                        beit_transforms=False,
                                        crop_scale=(1.0, 1.0),
                                        text_token_mask_prob=self.text_token_mask_prob,)

    def set_test_dataset(self):
        self.test_dataset = COCOCaptions(data_path=self.data_path,
                                         split='test',
                                         num_max_bpe_tokens=self.num_max_bpe_tokens,
                                         task=self.task,
                                         color_jitter=False,
                                         beit_transforms=False,
                                         crop_scale=(1.0, 1.0),
                                         text_token_mask_prob=0.0,)
        

class ConceptualCaptionsDataModule(BaseDataModule):
    def __init__(self,
                type,
                data_path,
                num_max_bpe_tokens,
                color_jitter=None,
                beit_transforms=False,
                crop_scale=(0.6, 1.0),
                text_token_mask_prob=0.0,
                *args,
                **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.type = type
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.color_jitter = color_jitter
        self.beit_transforms = beit_transforms
        self.crop_scale = crop_scale
        self.text_token_mask_prob = text_token_mask_prob

    def prepare_data(self):
        if not hasattr(self, 'train_dataset'):
            self.set_train_dataset()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = ConceptualCaptions(
            type=self.type,
            data_path=self.data_path,
            split='train',
            num_max_bpe_tokens=self.num_max_bpe_tokens,
            color_jitter=self.color_jitter,
            beit_transforms=self.beit_transforms,
            crop_scale=self.crop_scale,
            text_token_mask_prob=self.text_token_mask_prob,)


@register_datamodule(name='Flickr30k')
class Flickr30DataModule(BaseDataModule):
    def __init__(self,
                 data_path,
                 num_max_bpe_tokens,
                 color_jitter=None,
                 beit_transforms=False,
                 crop_scale=(0.6, 1.0),
                 text_token_mask_prob=0.0,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.color_jitter = color_jitter
        self.beit_transforms = beit_transforms
        self.crop_scale = crop_scale
        self.text_token_mask_prob = text_token_mask_prob

    def set_train_dataset(self):
        self.train_dataset = Flickr30Dataset(data_path=self.data_path,
                                             split='train',
                                             num_max_bpe_tokens=self.num_max_bpe_tokens,
                                             color_jitter=self.color_jitter,
                                             beit_transforms=self.beit_transforms,
                                             crop_scale=self.crop_scale,
                                             text_token_mask_prob=self.text_token_mask_prob,)

    def set_val_dataset(self):
        self.val_dataset = Flickr30Dataset(data_path=self.data_path,
                                           split='val',
                                           num_max_bpe_tokens=self.num_max_bpe_tokens,
                                           color_jitter=False,
                                           beit_transforms=False,
                                           crop_scale=(1.0, 1.0),
                                           text_token_mask_prob=self.text_token_mask_prob,)

    def set_test_dataset(self):
        self.test_dataset = Flickr30Dataset(data_path=self.data_path,
                                            split='test',
                                            num_max_bpe_tokens=self.num_max_bpe_tokens,
                                            color_jitter=False,
                                            beit_transforms=False,
                                            crop_scale=(1.0, 1.0),
                                            text_token_mask_prob=self.text_token_mask_prob,)


@register_datamodule(name="ConceptualCaptions3m")
def conceptual_captions_cc3m(*args, **kwargs):
    return ConceptualCaptionsDataModule(*args, type="cc3m", **kwargs)

@register_datamodule(name="ConceptualCaptions12m")
def conceptual_captions_cc12m(*args, **kwargs):
    return ConceptualCaptionsDataModule(*args, type="cc12m", **kwargs)
