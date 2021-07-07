import os
import hydra
import logging
import transformers as ppb
from BERT.bert_separated import fit_bert_separated


@hydra.main(config_path=r'configs\bert', config_name='bert_separated')
def run(cfg):
    logger = logging.getLogger(__name__)

    cfg_bert = cfg.bert
    cfg_datasets = cfg.datasets
    cfg_aug = cfg.augmentations
    cfg_loaders = cfg.loaders
    cfg_logreg = cfg.logreg

    data_path = cfg_datasets['data_path']
    train_path = os.path.join(data_path, cfg_datasets['train_filename'])
    val_path = os.path.join(data_path, cfg_datasets['val_filename'])
    test_path = os.path.join(data_path, cfg_datasets['test_filename'])

    logger.info('Evaluating pretrained BERT embeddings...')

    fit_bert_separated(logger=logger,
                       model_class=eval(cfg_bert['model_class']),
                       tokenizer_class=eval(cfg_bert['tokenizer_class']),
                       pretrained_weights=cfg_bert['pretrained_weights'],
                       train_path=train_path,
                       val_path=val_path,
                       test_path=test_path,
                       cfg_datasets=cfg_loaders,
                       prefixes=cfg_bert['prefixes'],
                       use_pooling=cfg_bert['use_pooling'],
                       cache_dir=cfg_datasets['cache_dir'],
                       augment=cfg_aug['augment'],
                       aug_steps=cfg_aug['aug_steps'],
                       enable_passage_aug=cfg_aug['enable_passage_aug'],
                       aug_batch_size=cfg_aug['aug_batch_size'],
                       penalty=cfg_logreg['penalty'],
                       l1_ratio=cfg_logreg['l1_ratio'],
                       solver=cfg_logreg['solver'],
                       max_iter=cfg_logreg['max_iter'])

    logger.info('Done')


if __name__ == '__main__':
    run()