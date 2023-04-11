from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale(X_train_num, X_val_num):
    scaler = StandardScaler()

    # Fit scaler to training data
    scaler.fit(X_train_num)

    # Scale training and validation data
    X_train_num = pd.DataFrame(scaler.transform(X_train_num), columns = X_train_num.columns, index = X_train_num.index)
    X_val_num = pd.DataFrame(scaler.transform(X_val_num), columns = X_val_num.columns, index = X_val_num.index)

    return X_train_num, X_val_num