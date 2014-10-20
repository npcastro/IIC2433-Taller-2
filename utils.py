from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

eros_filename = 'xm_visual_xtr.csv'

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', normed = False):
    """
    Recibe los arreglos de las clases correctas y hace un plot de la matriz de
    confusion usando colores
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # if normed:
    #     cm = cm.astype(float)
    #     for i in xrange(len(cm)):
    #         col = cm[:,i]
    #         suma = np.sum(col)
            
    #         if suma != 0:
    #             cm[:,i] = col/ float(suma)
    #         else:
    #             cm[:,i] = np.zeros(cm[:,i].shape)
    if normed:
        cm = cm.astype(float)
        for i in xrange(len(cm)):
            row = cm[i,:]
            suma = np.sum(row)
            
            if suma != 0:
                cm[i,:] = row/ float(suma)
            else:
                cm[i,:] = np.zeros(cm[i,:].shape)

    # Show confusion matrix in a separate window
    plt.matshow(cm)
    plt.title(title)
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def open_eros_data(path=eros_filename):
    eros_data = pd.read_csv(path)
    y = eros_data['type']
    del eros_data['type']
    del eros_data['#EROS_ID']
    X = eros_data
    return X, y

