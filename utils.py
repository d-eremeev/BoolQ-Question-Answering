from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd


def get_roc_auc(y_true,
                y_score,
                image_save_path=None):

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    preds_df = pd.DataFrame(data={'y_true': y_true, 'y_score': y_score})
    preds_df['y_true'] = preds_df['y_true'].astype(int)
    preds_df.to_csv('preds.csv', index=False)

    roc_df = pd.DataFrame(data={'fpr': fpr, 'tpr': tpr, 'threshold': thresholds})
    roc_df.to_csv('ROC.csv', index=False)

    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 1.5
    plt.plot(fpr, tpr, color='cornflowerblue',
             lw=lw, label='area = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc="lower right")

    if image_save_path is None:
        plt.show()
    else:
        plt.savefig(image_save_path)

    return roc_auc
