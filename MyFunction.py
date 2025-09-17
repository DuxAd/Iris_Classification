from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

    
def AffichageRes(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
        
    # Évaluer les performances du modèle
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    scores = cross_val_score(model, X_train, y_train, cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42))


    print('\n--------------------------------------------')
    print("\nRésultats du modele", model.__class__.__name__)
    print(f'Accuracy : {accuracy:.3f}, Precision : {precision:.3f}, recall :{recall:.3f}, F1 :{f1:.3f}')
    print(f'ROC_AUC : {roc_auc:.3f}')
    print('Score de validation Croisée')
    print(f'Score : {np.round(scores,3)}')
    print(f"Accuracy moyenne : {scores.mean():.3f}")
    print(f"Écart-type : {scores.std():.3}")
    print('\n--------------------------------------------\n')

    label = y_train.unique()
    cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                  normalize = 'true',
                                  #title = "Confusion Matrix "
                                  display_labels = label)

    #cm_display.plot()
    ax = plt.gca() # GCA = Get Current Axes
    ax.set_title(f"Matrice de Confusion de {model.__class__.__name__}")

    plt.show()
