from datasets import load_dataset, Features, Value
from functools import partial
from itertools import chain
import nltk
import nlpaug.augmenter.word as naw


def tokenize(x,
             tokenizer,
             prefixes,
             tokenization_type):
    """
    Function for tokenization
    - tokenization_type == 'separated': separately encode both question and passage elements of dataset.
      Both resulting sequences will have [CLS] and [SEP] tokens
    - tokenization_type == 'paired': encode question and passage as pairs.
      Resulting sequence will have 1 [CLS] and 2 [SEP] tokens:
      additional [SEP] will split question/passage in the resulting sequence.
    """

    if tokenization_type == 'separated':
        # input_ids, token_type_ids, attention_mask
        # token_type_ids == 0 when only 1 sequence passed
        tokenized_question = tokenizer(x['question'],
                                       padding='max_length',  # pad to max len for mini-batch learning
                                       truncation=True)

        tokenized_passage = tokenizer(x['passage'],
                                      padding='max_length',  # pad to max len
                                      truncation=True)

        tokenized_question = {f'{prefixes["question"]}_{k}':v for k,v in tokenized_question.items()}
        tokenized_passage = {f'{prefixes["passage"]}_{k}':v for k,v in tokenized_passage.items()}
        out_dict = {**tokenized_question, **tokenized_passage}

        out_dict['answer'] = x['answer']
    elif tokenization_type == 'paired':
        # https://discuss.huggingface.co/t/use-two-sentences-as-inputs-for-sentence-classification/5444/4
        # https://huggingface.co/transformers/main_classes/tokenizer.html
        tokenized_pair = tokenizer(text=x['question'],
                                   text_pair=x['passage'],
                                   truncation=True,
                                   padding='max_length'  # pad to max len
                                   )

        out_dict = dict(tokenized_pair)
    else:
        raise NotImplementedError

    return out_dict


# https://github.com/huggingface/datasets/issues/365
def text_augment(samples,
                 question_aug,
                 passage_aug,
                 aug_steps=3,
                 enable_passage_aug=False):

    # samples here is dict of lists
    # we are doing different augmentations for different keys
    # SynonymAug for questions
    # BackTranslationAug for passages

    if aug_steps == 0:
        return samples

    init_answers = samples['answer']
    init_questions = samples['question']
    init_passages = samples['passage']

    augmented_questions = []
    for _ in range(aug_steps):
        augmented_questions.append(question_aug.augment(init_questions))

    if enable_passage_aug:
        augmented_passage = passage_aug.augment(init_passages)  # apply back translation and replicate
        augmented_passage = augmented_passage.copy() * aug_steps
    else:
        augmented_passage = init_passages.copy() * aug_steps  # just replicate initial passages

    augmented_questions = list(chain(*augmented_questions))

    samples['answer'] = init_answers + init_answers * aug_steps
    samples['question'] = init_questions + augmented_questions
    samples['passage'] = init_passages + augmented_passage

    return samples


def get_dataset(logger,
                train_path,
                val_path,
                test_path,
                tokenizer,
                encode_type,
                cache_dir,
                prefixes=None,
                augment=False,
                aug_steps=None,
                enable_passage_aug=False,
                aug_batch_size=16
                ):
    """
    Loads datasets, applies tokenization
    """

    if encode_type == 'separated' and prefixes is None:
        raise ValueError('You should pass question/passage prefixes when encode_type = separated')

    features = Features({
        "question": Value("string"),
        "passage": Value("string"),
        "answer": Value("int32")
    })

    dataset = load_dataset('csv',
                           data_files={'train': train_path, 'valid': val_path, 'test': test_path},
                           features=features,
                           cache_dir=cache_dir)

    if augment:
        stop_words = nltk.corpus.stopwords.words('english')
        logger.info('Question Synonym Aug enabled')
        aug_synonym = naw.SynonymAug(aug_src='wordnet', stopwords=stop_words)
        if enable_passage_aug:
            logger.info('Passage Back Translation enabled')
            back_translation_aug = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', #https://huggingface.co/facebook/wmt19-en-de
                                                          to_model_name='facebook/wmt19-de-en',
                                                          device='cuda')
        else:
            back_translation_aug = None

        text_augment_f = partial(text_augment,
                                 question_aug=aug_synonym,
                                 passage_aug=back_translation_aug,
                                 aug_steps=aug_steps,
                                 enable_passage_aug=enable_passage_aug)

        logger.info('Starting augmentation process')
        dataset['train'] = dataset['train'].map(text_augment_f,
                                                batched=True,
                                                batch_size=aug_batch_size)

        logger.info('Augmentations completed')

    tokenize_f = partial(tokenize,
                         tokenizer=tokenizer,
                         prefixes=prefixes,
                         tokenization_type=encode_type)

    dataset = dataset.map(tokenize_f,
                          batched=False)  # True doesn't work in paired case

    # below we define list of used dataset fields in each case
    used_cols = ['answer']

    if encode_type == 'separated':
        for el in ['attention_mask', 'input_ids']:
            for prefix in prefixes.values():
                used_cols.append(f'{prefix}_{el}')
    elif encode_type == 'paired':
        used_cols = used_cols + ['attention_mask', 'input_ids'] + ['token_type_ids']
    else:
        raise NotImplementedError

    dataset.set_format(type='torch', columns=used_cols)

    return dataset
