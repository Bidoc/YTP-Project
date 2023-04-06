import os
import time
import pandas as pd
import numpy as np
import IPython as ipd
import librosa 
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from datetime import datetime
from sklearn import metrics
from tqdm import tqdm
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
audio_dataset_path='C:/Users/Tran Pham Gia Bao/Python/Data/genres_original'
metadata=pd.read_csv('C:/Users/Tran Pham Gia Bao/Python/Data/features_30_sec.csv')
metadata.head()
def feature_extractor(file):
    audio, sample_rate=librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features=librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features=np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features
metadata.drop(labels=552, axis=0, inplace=True)
extracted_features=[]
for index_num, row in tqdm(metadata.iterrows()):
    try:
        final_class_labels=row["label"]
        file_name=os.path.join(os.path.abspath(audio_dataset_path), final_class_labels+'/', str(row["filename"]))
        data=feature_extractor(file_name)
        extracted_features.append([data, final_class_labels])
    except Exception as e:
        print(f"Error: {e}")
        continue
extracted_features_df=pd.DataFrame(extracted_features, columns=['features', 'class'])
extracted_features_df.head()
X=np.array(extracted_features_df['features'].tolist())
y=np.array(extracted_features_df['class'].to_list())
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)
num_labels=y.shape[1]
model=Sequential()
model.add(Dense(1024, input_shape=(40,), activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(num_labels, activation="softmax"))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
t=time.localtime()
current_time=time.strftime("%H.%M.%S", t)

num_epochs=100
num_batch_size=32

checkpointer=ModelCheckpoint(filepath=f'save_modes/audio_classification_{current_time}.hdf5', verbose=1, save_best_only=True)

start=datetime.now()

history=model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)

duration=datetime.now()-start
print("Training completed in time: ", duration)

model.evaluate(X_test, y_test, verbose=0)
pd.DataFrame(history.history).plot(figsize=(12,6))
plt.show()
model.save(filepath=f'save_modes/audio_classification_{current_time}.hdf5')
#model.predict_classes(X_test)
