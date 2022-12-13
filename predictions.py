from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

def predict(model, X):
    arr_X= np.array(X)
    # scaler = StandardScaler()
    # scaled_X=scaler.fit_transform(X)
    pred_X= classifier_model.predict(arr_X)
    # pred_X = scaler.inverse_transform(pred_X)
    print(pred_X.shape)
    
    return pred_X

def get_prediction_list(pred_X):
    predictions = []
    for block in pred_X:
        indices = np.argmax(block)
        predictions.append(indices)


    print(len(predictions))
    print(np.unique(predictions))
    
    return predictions

def get_original_labels(Y):
    arr_Y = np.array(Y)
    print(arr_Y.shape)

    orig_labels = []
    for block in arr_Y:
        indices = np.argmax(block)
        orig_labels.append(indices)

    print(len(orig_labels))
    print(np.unique(orig_labels))
    return orig_labels

def get_classification_report(orig_labels, predictions, data_labels, filename):
    report = (classification_report(orig_labels, predictions, target_names=data_labels, output_dict= True))
    print(report)

    report_df = pd.DataFrame(report)
    print(report_df)

    report_df.to_csv('controlled_report.csv')
    
    return report,  report_df

def get_confusion_matrix(orig_labels, predictions, data_labels):
    cm =confusion_matrix(orig_labels, predictions)#, normalize= 'true'
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels =data_labels)
    disp.plot()

    cm_values = confusion_matrix(orig_labels, predictions).ravel()
    print(cm_values)
    
    return cm, cm_values