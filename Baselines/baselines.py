import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from nltk import WordNetLemmatizer, WordPunctTokenizer
from nltk.corpus import stopwords
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
import logging
import gensim.models.word2vec
import gensim.models.fasttext
import gensim.utils
from utils import get_roc_auc

logging.getLogger("gensim.models.word2vec").setLevel(logging.ERROR)
logging.getLogger("gensim.models.fasttext").setLevel(logging.ERROR)
logging.getLogger("gensim.models.base_any2vec").setLevel(logging.ERROR)
logging.getLogger("gensim.utils").setLevel(logging.ERROR)

class ConstantBaseline():
    """
    Assigns most frequent answer to every question
    """
    def __init__(self,
                 logger):

        self.logger = logger

    def fit(self,
            train_path,
            val_path,
            test_path):
        """
        Assigns the most frequent answer on Train to each question in Test part
        """
        train, val, test = self.load_data(train_path=train_path,
                                          val_path=val_path,
                                          test_path=test_path)

        self.logger.info('Data has been loaded')
        self.logger.info(f'Train shape: {train.shape}, Val shape: {val.shape}, Test shape: {test.shape}')

        most_frequent_ans = train.answer.mode()
        assert len(most_frequent_ans) == 1
        most_frequent_ans = most_frequent_ans.item()
        test['preds'] = most_frequent_ans

        acc = accuracy_score(test['answer'], test['preds'])
        self.logger.info(f'Constant Baseline Accuracy (Test): {acc}')

    def load_data(self,
                  train_path,
                  val_path,
                  test_path):
        """
        Loads jsons to dict of Pandas DataFrames
        """
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)

        return train, val, test

class FastTextBaseline():
    """
    Learns FastText embeddings on concatenated question-passage pairs.
    Probability of "True" answer is given by Logistic Regression,
    fitted on averaged embeddings for tokens in each pair.
    """
    def __init__(self,
                 logger):

        self.tokenizer = WordPunctTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.logger = logger

    def load_data(self,
                  train_path,
                  val_path,
                  test_path):
        """
        Loads jsons to dict of Pandas DataFrames
        """
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        test= pd.read_csv(test_path)
        return train, val, test

    def fit(self,
            train_path,
            val_path,
            test_path,
            lemmatization=True,
            remove_stopwords=False,
            epochs=100,
            early_stopping_rounds=25,
            model_prefix='fasttext',
            track_loss=False,
            vector_size=32,
            window=25,
            alpha=0.05,
            min_alpha=0.05,
            min_count=1,
            penalty='l2',
            l1_ratio=None,
            solver='lbfgs',
            max_iter=100
            ):
        """
        Learns FastText unsupervised embeddings on concatenated question-passage pair.
        Num of epochs is controlled by Downstream Task accuracy score:
        Binary Classification of answers on "dev" part.
        """
        train, val, test = self.load_data(train_path=train_path,
                                          val_path=val_path,
                                          test_path=test_path)

        self.logger.info('Data has been loaded')

        self.logger.info(f'Number of train examples: {train.shape[0]}')
        self.logger.info(f'Number of valid examples: {val.shape[0]}')
        self.logger.info(f'Number of test examples: {val.shape[0]}')

        self.logger.info(f'Lemmatization set to {lemmatization}, remove_stopwords set to {remove_stopwords}')

        self.prepare_datasets(dfs=[train, val, test],
                              lemmatization=lemmatization,
                              remove_stopwords=remove_stopwords)

        self.logger.info(f'Pairs concatenated and tokenized.')

        model_saver = ModelSaver(df_train=train,
                                 df_val=val,
                                 model_prefix=model_prefix,
                                 early_stop_rounds=early_stopping_rounds,
                                 logger=self.logger,
                                 penalty=penalty,
                                 l1_ratio=l1_ratio,
                                 solver=solver,
                                 max_iter=max_iter)

        epoch_logger = EpochLogger(logger=self.logger)

        callback_list = [model_saver, epoch_logger]

        if track_loss:
            loss_logger = LossLogger(self.logger)
            callback_list.append(loss_logger)

        self.logger.info('')
        self.logger.info('Starting to train FASTTEXT')

        # TRAIN UNSUPERVISED FASTTEXT
        model = FastText(vector_size=vector_size,
                         window=window,
                         alpha=alpha,
                         min_alpha=min_alpha,
                         min_count=min_count)

        model.build_vocab(corpus_iterable=train['tokens'])
        total_examples = model.corpus_total_words
        total_words = model.corpus_total_words

        try:
            model.train(corpus_iterable=train['tokens'],
                        total_examples=total_examples,
                        total_words=total_words,
                        epochs=epochs,
                        compute_loss=True,
                        callbacks=callback_list)
        except EarlyStoppingException:
            self.logger.info('Met Early Stopping Condition! Training aborted.')

        self.logger.info('')
        self.logger.info('Loading Best model')

        # LOAD BEST MODEL
        model = FastText.load(f'{model_prefix}.model')

        self.logger.info('Inference on Test')

        # INFERENCE ON TEST
        acc, score, _, _ = FastTextBaseline.fitLogReg(emb_model=model,
                                                      df_train=train,
                                                      df_val=test,
                                                      penalty=penalty,
                                                      l1_ratio=l1_ratio,
                                                      solver=solver,
                                                      max_iter=max_iter,
                                                      scale=True)

        roc_auc = get_roc_auc(y_true=test['answer'].values,
                              y_score=score,
                              image_save_path='roc.jpg')

        self.logger.info(f'LogReg on FastText embeddings Accuracy (Test): {round(acc, 3)}')
        self.logger.info(f'LogReg on FastText embeddings ROC-AUC (Test): {round(roc_auc, 3)}')

    def prepare_datasets(self,
                         dfs,
                         lemmatization,
                         remove_stopwords):
        """
        Adds column 'tokens' that contains preprocessed  [question, passage] concatenated strings
        """
        for df in dfs:
            df['tokens'] = df[['question', 'passage']].apply(lambda x: ' '.join(x), axis=1)
            df['tokens'] = df['tokens'].apply(self.tokenize,
                                              lemmatization=lemmatization,
                                              remove_stopwords=remove_stopwords)

    @staticmethod
    def tokens2vec(tokens,
                   emb_model):
        """
        Averages across list of tokens embeddings
        """
        dim = emb_model.vector_size
        tokens_embedding = np.zeros(dim, dtype=np.float32)
        word_counter = 0
        for token in tokens:
            tokens_embedding += emb_model.wv[token]
            word_counter += 1
        if word_counter != 0:
            tokens_embedding /= word_counter
        return tokens_embedding

    def tokenize(self,
                 sent,
                 lemmatization=True,
                 remove_stopwords=True):
        """
        Returns a list of tokens for a given string (sent).
        Applies lemmatization and removes stopwords if required.
        """
        tokens = self.tokenizer.tokenize(sent.lower())
        if remove_stopwords:
            tokens = [el for el in tokens if el not in self.stopwords]
        if lemmatization:
            tokens = [self.lemmatizer.lemmatize(el) for el in tokens]
        return tokens

    @staticmethod
    def fitLogReg(emb_model,
                  df_train,
                  df_val,
                  penalty,
                  l1_ratio,
                  solver,
                  max_iter,
                  scale=True):
        """
        Fits Logistic Regression on averaged fasttext embeddings (of tokens in [question, passage])
        """
        scaler = None

        X_train = np.array(df_train['tokens'].apply(FastTextBaseline.tokens2vec, emb_model=emb_model).tolist())
        X_val = np.array(df_val['tokens'].apply(FastTextBaseline.tokens2vec, emb_model=emb_model).tolist())
        y_train = df_train['answer'].values
        y_val = df_val['answer'].values

        if scale:
            scaler = StandardScaler().fit(X_train)
            X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)

        clf = LogisticRegression(random_state=0,
                                 penalty=penalty,
                                 l1_ratio=l1_ratio,
                                 solver=solver,
                                 max_iter=max_iter
                                 ).fit(X_train, y_train)

        score = clf.predict_proba(X_val)[:, 1]
        acc = ((score >= 0.5) == y_val).sum() / score.shape[0]
        return acc, score, clf, scaler


class EarlyStoppingException(Exception):
    pass


class ModelSaver(CallbackAny2Vec):
    """
    Callback to save model based on downstream-task Accuracy
    """

    def __init__(self,
                 df_train,
                 df_val,
                 model_prefix,
                 early_stop_rounds,
                 logger,
                 penalty,
                 l1_ratio,
                 solver,
                 max_iter):

        self.model_prefix = model_prefix
        self.best_downstream_acc = 0
        self.early_stop_rounds = early_stop_rounds
        self.epochs_without_improvement = 0
        self.logger = logger
        self.df_train = df_train
        self.df_val = df_val
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.max_iter = max_iter

    def on_epoch_end(self, model):
        downstream_acc, _, _, _ = FastTextBaseline.fitLogReg(emb_model=model,
                                                             df_train=self.df_train,
                                                             df_val=self.df_val,
                                                             penalty=self.penalty,
                                                             l1_ratio=self.l1_ratio,
                                                             solver=self.solver,
                                                             max_iter=self.max_iter,
                                                             scale=True)

        self.logger.info(f'Downstream Accuracy = {round(downstream_acc, 3)}')

        if downstream_acc > self.best_downstream_acc:
            self.epochs_without_improvement = 0
            self.best_downstream_acc = downstream_acc
            output_path = f'{self.model_prefix}.model'
            model.save(output_path)
            self.logger.info(f'Model saved!')
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.early_stop_rounds:
                raise EarlyStoppingException()


class EpochLogger(CallbackAny2Vec):
    """
    Callback to log information about training
    """

    def __init__(self,
                 logger):
        self.epoch = 1
        self.logger_ = logger

    def on_epoch_begin(self, model):
        self.logger_.info('')
        self.logger_.info(f"Epoch #{self.epoch} start")

    def on_epoch_end(self, model):
        self.logger_.info(f"Epoch #{self.epoch} end")
        self.epoch += 1


class LossLogger(CallbackAny2Vec):
    """
    Callback to print loss after each epoch.
    """
    def __init__(self,
                 logger):
        self.epoch = 1
        self.loss_to_be_subed = 0
        self.logger = logger

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        self.logger.info(f'FastText_Loss after epoch {self.epoch}: {loss_now}')
        self.epoch += 1



