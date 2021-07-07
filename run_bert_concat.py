import os
import hydra
import logging
import transformers as ppb
from BERT.bert_concat import train_bert_concat


@hydra.main(config_path=r'configs\bert', config_name='bert_concat')
def run(cfg):
    logger = logging.getLogger(__name__)

    cfg_bert = cfg.bert
    cfg_datasets = cfg.datasets
    cfg_loaders = cfg.loaders
    cfg_aug = cfg.augmentations

    data_path = cfg_datasets['data_path']
    train_path = os.path.join(data_path, cfg_datasets['train_filename'])
    val_path = os.path.join(data_path, cfg_datasets['val_filename'])
    test_path = os.path.join(data_path, cfg_datasets['test_filename'])

    logger.info('Fine-tuning BERT ...')

    train_bert_concat(logger=logger,
                      model_class=eval(cfg_bert['model_class']),
                      hidden_dropout_prob=cfg_bert['hidden_dropout_prob'],
                      tokenizer_class=eval(cfg_bert['tokenizer_class']),
                      pretrained_weights=cfg_bert['pretrained_weights'],
                      train_path=train_path,
                      val_path=val_path,
                      test_path=test_path,
                      cfg_datasets=cfg_loaders,
                      freeze_bert=cfg_bert['freeze_bert'],
                      epochs=cfg_bert['epochs'],
                      lr=cfg_bert['lr'],
                      cache_dir=cfg_datasets['cache_dir'],
                      augment=cfg_aug['augment'],
                      aug_steps=cfg_aug['aug_steps'],
                      enable_passage_aug=cfg_aug['enable_passage_aug'],
                      aug_batch_size=cfg_aug['aug_batch_size'])

    logger.info('Done')

if __name__ == '__main__':
    run()