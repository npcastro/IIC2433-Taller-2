from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

eros_filename = 'xm_visual_xtr.csv'

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    Recibe los arreglos de las clases correctas y hace un plot de la matriz de
    confusion usando colores
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

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

