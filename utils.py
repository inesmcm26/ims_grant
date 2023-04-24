from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

def scale(X_train_num, X_val_num):
    # Use StandardScaler for gaussian features
    gaussian_feats = ['PPEDUC','DISTRESS','MATERIALISM_1','MATERIALISM_2','ASK1_2','FSscore','FWBscore']

    scaler = StandardScaler()
    # Fit scaler to training data
    scaler.fit(X_train_num[gaussian_feats])
    # Scale training and validation data
    gaussian_scaled_train = pd.DataFrame(scaler.transform(X_train_num[gaussian_feats]), columns = gaussian_feats, index = X_train_num.index)
    gaussian_scaled_val = pd.DataFrame(scaler.transform(X_val_num[gaussian_feats]), columns = gaussian_feats, index = X_val_num.index)

    # Use MinMaxScaler for non-gaussian features
    non_gaussian = list(set(X_train_num.columns) - set(['PPEDUC','DISTRESS','MATERIALISM_1','MATERIALISM_2','ASK1_2','FSscore','FWBscore']))

    scaler = MinMaxScaler()
    # Fit scaler to training data
    scaler.fit(X_train_num[non_gaussian])
    # Scale training and validation data
    non_gaussian_scaled_train = pd.DataFrame(scaler.transform(X_train_num[non_gaussian]), columns = non_gaussian, index = X_train_num.index)
    non_gaussian_scaled_val = pd.DataFrame(scaler.transform(X_val_num[non_gaussian]), columns = non_gaussian, index = X_val_num.index)

    X_train_num = pd.concat([gaussian_scaled_train, non_gaussian_scaled_train], axis = 1)
    X_val_num = pd.concat([gaussian_scaled_val, non_gaussian_scaled_val], axis = 1)

    return X_train_num, X_val_num