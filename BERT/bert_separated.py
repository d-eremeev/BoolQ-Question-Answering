import torch
import numpy as np
from torch.utils.data import DataLoader
from BERT.dataset import get_dataset
from tqdm import tqdm
from BERT.models import PretrainedBERT
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils import get_roc_auc


def fit_bert_separated(logger,
                       model_class,
                       tokenizer_class,
                       pretrained_weights,
                       train_path,
                       val_path,
                       test_path,
                       cfg_datasets,
                       prefixes,
                       use_pooling,
                       cache_dir,
                       augment,
                       aug_steps,
                       enable_passage_aug,
                       aug_batch_size,
                       penalty='l2',
                       l1_ratio=None,
                       solver='lbfgs',
                       max_iter=100):
    """
    - Loads datasets.
    - Loads  pretrained BERT model and tokenizer.
    - Concatenates BERT embeddings for tokenized questions and passages.
    - If 'use_pooling=False' - uses concatenation of [CLS] states. Else - concatenation of pooling vectors.
    - Fits Logistic regression on obtained feature-vectors.
    """

    model_class, tokenizer_class, pretrained_weights = (model_class,
                                                        tokenizer_class,
                                                        pretrained_weights)

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    logger.info('Loading datasets...')

    dataset = get_dataset(logger=logger,
                          train_path=train_path,
                          val_path=val_path,
                          test_path=test_path,
                          tokenizer=tokenizer,
                          encode_type='separated',
                          prefixes=prefixes,
                          cache_dir=cache_dir,
                          augment=augment,
                          aug_steps=aug_steps,
                          enable_passage_aug=enable_passage_aug,
                          aug_batch_size=aug_batch_size
                          )

    logger.info(f"Number of train examples: {len(dataset['train'])}")
    logger.info(f"Number of valid examples: {len(dataset['valid'])}")
    logger.info(f"Number of test examples: {len(dataset['test'])}")

    logger.info('Creating dataloaders...')
    train_loader = DataLoader(dataset['train'],
                              batch_size=cfg_datasets['train']['batch_size'],
                              shuffle=cfg_datasets['train']['shuffle'],
                              num_workers=cfg_datasets['train']['num_workers'])

    test_loader = DataLoader(dataset['test'],
                             batch_size=cfg_datasets['test']['batch_size'],
                             shuffle=cfg_datasets['test']['shuffle'],
                             num_workers=cfg_datasets['test']['num_workers'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_ = PretrainedBERT(pretrained_bert=model,
                            use_pooling=use_pooling)

    model_ = model_.to(device)

    params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model_.parameters())])
    logger.info(f"Number of model params: {params}")

    logger.info('Inference BERT on train/test...')

    train_vectors, train_labels = inference(loader=train_loader,
                                            model_=model_,
                                            device=device,
                                            prefixes=prefixes)

    test_vectors, test_labels = inference(loader=test_loader,
                                          model_=model_,
                                          device=device,
                                          prefixes=prefixes)

    logger.info(f'Fitting LogReg on BERT embeddings of question/passage pairs...')

    acc, score, _, _ = fitLogReg(train_vectors=train_vectors,
                                 test_vectors=test_vectors,
                                 train_labels=train_labels,
                                 test_labels=test_labels,
                                 penalty=penalty,
                                 l1_ratio=l1_ratio,
                                 solver=solver,
                                 max_iter=max_iter,
                                 scale=True)

    roc_auc = get_roc_auc(y_true=test_labels,
                          y_score=score,
                          image_save_path='roc.jpg')

    logger.info(f'Accuracy (Test): {round(acc, 3)}')
    logger.info(f'ROC-AUC (Test): {round(roc_auc, 3)}')


@torch.no_grad()
def inference(loader,
              model_,
              device,
              prefixes):

    model_.eval()

    pbar = tqdm(enumerate(loader), total=len(loader))
    pbar.set_description(f"Inference: ")

    outputs = []
    labels = []
    for i, batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        output = model_(tokens_q=batch[f'{prefixes["question"]}_input_ids'],
                        tokens_p=batch[f'{prefixes["passage"]}_input_ids'],
                        att_mask_q=batch[f'{prefixes["question"]}_attention_mask'],
                        att_mask_p=batch[f'{prefixes["passage"]}_attention_mask'])

        outputs.append(output)
        labels.append(batch['answer'])

    outputs = torch.cat(outputs)
    labels = torch.cat(labels)

    return outputs.cpu().numpy(), labels.cpu().numpy()


def fitLogReg(train_vectors,
              test_vectors,
              train_labels,
              test_labels,
              penalty,
              l1_ratio,
              solver,
              max_iter,
              scale=True
              ):
    """
    Fits Logistic Regression on train_vectors, infers it on test_vectors
    """

    scaler = None
    if scale:
        scaler = StandardScaler().fit(train_vectors)
        train_vectors, test_vectors = scaler.transform(train_vectors), scaler.transform(test_vectors)

    clf = LogisticRegression(random_state=0,
                             penalty=penalty,
                             l1_ratio=l1_ratio,
                             solver=solver,
                             max_iter=max_iter).fit(train_vectors, train_labels)

    score = clf.predict_proba(test_vectors)[:, 1]
    acc = ((score >= 0.5) == test_labels).sum() / score.shape[0]
    return acc, score, clf, scaler
