import torch
import numpy as np
from torch.utils.data import DataLoader
from BERT.dataset import get_dataset
from tqdm import tqdm
from transformers import AdamW
import torch.nn.functional as F
from utils import get_roc_auc


def train_bert_concat(logger,
                      model_class,
                      hidden_dropout_prob,
                      tokenizer_class,
                      pretrained_weights,
                      train_path,
                      val_path,
                      test_path,
                      cfg_datasets,
                      freeze_bert,
                      epochs,
                      lr,
                      cache_dir,
                      augment,
                      aug_steps,
                      enable_passage_aug,
                      aug_batch_size
                      ):
    """
    Trains (fine-tuning) BertForSequenceClassification: bert with sequence classification/regression head on top of pooled output.
    https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification

    Tokenized question/passage pairs serve as sequences, separated by [SEP] token.
    """

    model_class, tokenizer_class, pretrained_weights = (model_class,
                                                        tokenizer_class,
                                                        pretrained_weights)

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights,
                                        hidden_dropout_prob=hidden_dropout_prob)

    logger.info('Loading datasets...')

    dataset = get_dataset(logger=logger,
                          train_path=train_path,
                          val_path=val_path,
                          test_path=test_path,
                          tokenizer=tokenizer,
                          encode_type='paired',
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

    valid_loader = DataLoader(dataset['valid'],
                              batch_size=cfg_datasets['val']['batch_size'],
                              shuffle=cfg_datasets['val']['shuffle'],
                              num_workers=cfg_datasets['val']['num_workers'])

    test_loader = DataLoader(dataset['test'],
                             batch_size=cfg_datasets['test']['batch_size'],
                             shuffle=cfg_datasets['test']['shuffle'],
                             num_workers=cfg_datasets['test']['num_workers'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
    logger.info(f"Number of model initial params : {params}")

    # freeze all layers except "classifier" layer if required
    # (for ex. see https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForSequenceClassification)
    # https://github.com/huggingface/transformers/issues/400
    if freeze_bert:
        logger.info('Freezing BERT body')
        for name, param in model.named_parameters():
            if 'classifier' not in name:  # classifier layer
                param.requires_grad = False

    params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
    logger.info(f"Number of model params for training: {params}")

    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # https://huggingface.co/transformers/main_classes/optimizer_schedules.html
    # lr_scheduler = eval(cfg['lr_scheduler']['name'])(optimizer=optim,
    #                                                  num_warmup_steps=num_warmup_steps,
    #                                                  num_training_steps=num_training_steps)
    #
    # logger.info(f"LR scheduler: {lr_scheduler}")

    lowest_loss = np.inf

    #### TRAINING LOOP ####

    for epoch in range(epochs):
        model.train()

        # current_lr = lr_scheduler.optimizer.param_groups[0]['lr']
        # logger.info(f"Current learning rate: {current_lr}")

        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch}: ")

        # -- train --
        losses = 0
        for batch in pbar:
            optim.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}

            # check https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
            # for the output content
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            token_type_ids=batch['token_type_ids'],
                            labels=batch['answer'].long())

            loss = outputs.loss  # loss on current mini-batch
            loss.backward()
            optim.step()
            # lr_scheduler.step()

            losses += loss.item()

        train_loss = losses/len(train_loader)

        logger.info(f'Epoch {epoch}, Train Loss {round(train_loss, 4)}')

        # -- validate--
        val_metrics_dict = valid(valid_loader,
                                 dataset['valid'],
                                 model=model,
                                 device=device,
                                 inference=False)

        val_loss, val_accuracy = val_metrics_dict["loss"], val_metrics_dict["accuracy"]
        logger.info(f'Epoch {epoch}, Val Loss: {round(val_loss, 4)}, Val Accuracy: {round(val_accuracy, 4)}')

        # -- save model --
        if val_loss < lowest_loss:
            torch.save(model.state_dict(), 'model_qa.pt')
            lowest_loss = val_loss
            logger.info(f'Model saved!')

    # -- test --
    model.load_state_dict(torch.load('model_qa.pt'))
    test_metrics_dict, test_scores, test_labels = valid(test_loader,
                                                        dataset['test'],
                                                        model=model,
                                                        device=device,
                                                        inference=True)
    test_loss, test_accuracy = test_metrics_dict["loss"], test_metrics_dict["accuracy"]

    logger.info(f'Test Loss: {round(test_loss, 3)}, Test Accuracy: {round(test_accuracy, 3)}')

    roc_auc = get_roc_auc(y_true=test_labels,
                          y_score=test_scores,
                          image_save_path='roc.jpg')

    logger.info(f'ROC-AUC (Test): {round(roc_auc, 3)}')

@torch.no_grad()
def valid(loader,
          dataset,
          model,
          device,
          inference=False):
    """
    Validation / Inference
    When performing inference on Test we should return scores - predicted probabilities of class 1 - for classification report
    """

    model.eval()

    pbar = tqdm(loader, total=len(loader))
    pbar.set_description(f"Inference: ")

    scores = []
    labels = []
    losses = 0
    accuracy = 0
    for batch in pbar:

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids'],
                        labels=batch['answer'].long())

        labels.append(batch['answer'])

        # outputs.logits - scores [batch_size, 2] before SoftMax
        # compute probabilities of class 1
        score = F.softmax(outputs.logits, dim=1)[:, 1]

        accuracy += ((score >= 0.5) == batch['answer']).sum().item()
        scores.append(score)
        loss = outputs.loss  # loss on current mini-batch
        losses += loss.item()

    val_loss = losses / len(loader)
    accuracy = accuracy / len(dataset)

    metrics_dict = {'accuracy': accuracy,
                    'loss': val_loss}

    scores = torch.cat(scores)
    labels = torch.cat(labels)

    if not inference:
        return metrics_dict
    else:
        return metrics_dict, scores.cpu().numpy(), labels.cpu().numpy()

