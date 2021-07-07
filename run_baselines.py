import hydra
import logging
from Baselines.baselines import ConstantBaseline, FastTextBaseline
import os


@hydra.main(config_path=r'configs\baselines', config_name='baselines')
def run(cfg):
    logger = logging.getLogger(__name__)

    cfg_datasets = cfg.datasets
    cfg_fasttext = cfg.fasttext
    cfg_logreg = cfg.logreg

    data_path = cfg_datasets['data_path']
    train_path = os.path.join(data_path, cfg_datasets['train_filename'])
    val_path = os.path.join(data_path, cfg_datasets['val_filename'])
    test_path = os.path.join(data_path, cfg_datasets['test_filename'])

    logger.info('Evaluating Constant baseline...')
    constantbaseline = ConstantBaseline(logger)

    constantbaseline.fit(train_path=train_path,
                         val_path=val_path,
                         test_path=test_path)
    logger.info('Done')

    logger.info('')

    logger.info('Evaluating FastText baseline...')
    fasttextbaseline = FastTextBaseline(logger)

    fasttextbaseline.fit(train_path=train_path,
                         val_path=val_path,
                         test_path=test_path,
                         lemmatization=cfg_fasttext['lemmatization'],
                         remove_stopwords=cfg_fasttext['remove_stopwords'],
                         epochs=cfg_fasttext['epochs'],
                         early_stopping_rounds=cfg_fasttext['early_stopping_rounds'],
                         model_prefix=cfg_fasttext['model_prefix'],
                         track_loss=cfg_fasttext['track_loss'],
                         vector_size=cfg_fasttext['vector_size'],
                         window=cfg_fasttext['window'],
                         alpha=cfg_fasttext['alpha'],
                         min_alpha=cfg_fasttext['min_alpha'],
                         min_count=cfg_fasttext['min_count'],
                         penalty=cfg_logreg['penalty'],
                         l1_ratio=cfg_logreg['l1_ratio'],
                         solver=cfg_logreg['solver'],
                         max_iter=cfg_logreg['max_iter'])

    logger.info('Done')

if __name__ == '__main__':
    run()