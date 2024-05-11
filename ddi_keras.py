import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import ClusterCentroids
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss 
from numpy import mean
from numpy import std
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import tensorflow as tf

data = pd.read_csv("nozeros2024.csv")
feature_cols=["rel1_ADMINISTERED_TO", "rel1_AFFECTS", "rel1_ASSOCIATED_WITH", "rel1_AUGMENTS", "rel1_CAUSES", "rel1_COEXISTS_WITH", "rel1_compared_with", "rel1_COMPLICATES", "rel1_CONVERTS_TO", "rel1_DIAGNOSES", "rel1_different_from", "rel1_different_than", "rel1_DISRUPTS", "rel1_higher_than", "rel1_INHIBITS", "rel1_INTERACTS_WITH", "rel1_IS_A", "rel1_ISA", "rel1_LOCATION_OF", "rel1_lower_than", "rel1_MANIFESTATION_OF", "rel1_METHOD_OF", "rel1_OCCURS_IN", "rel1_PART_OF", "rel1_PRECEDES", "rel1_PREDISPOSES", "rel1_PREVENTS", "rel1_PROCESS_OF", "rel1_PRODUCES", "rel1_same_as", "rel1_STIMULATES", "rel1_TREATS", "rel1_USES", "rel1_MENTIONED_IN", "rel1_HAS_MESH", "rel2_ADMINISTERED_TO", "rel2_AFFECTS", "rel2_ASSOCIATED_WITH", "rel2_AUGMENTS", "rel2_CAUSES", "rel2_COEXISTS_WITH", "rel2_compared_with", "rel2_COMPLICATES", "rel2_CONVERTS_TO", "rel2_DIAGNOSES", "rel2_different_from", "rel2_different_than", "rel2_DISRUPTS", "rel2_higher_than", "rel2_INHIBITS", "rel2_INTERACTS_WITH", "rel2_IS_A", "rel2_ISA", "rel2_LOCATION_OF", "rel2_lower_than", "rel2_MANIFESTATION_OF", "rel2_METHOD_OF", "rel2_OCCURS_IN", "rel2_PART_OF", "rel2_PRECEDES", "rel2_PREDISPOSES", "rel2_PREVENTS", "rel2_PROCESS_OF", "rel2_PRODUCES", "rel2_same_as", "rel2_STIMULATES", "rel2_TREATS", "rel2_USES", "rel2_MENTIONED_IN", "rel2_HAS_MESH", "rel3_ADMINISTERED_TO", "rel3_AFFECTS", "rel3_ASSOCIATED_WITH", "rel3_AUGMENTS", "rel3_CAUSES", "rel3_COEXISTS_WITH", "rel3_compared_with", "rel3_COMPLICATES", "rel3_CONVERTS_TO", "rel3_DIAGNOSES", "rel3_different_from", "rel3_different_than", "rel3_DISRUPTS", "rel3_higher_than", "rel3_INHIBITS", "rel3_INTERACTS_WITH", "rel3_IS_A", "rel3_ISA", "rel3_LOCATION_OF", "rel3_lower_than", "rel3_MANIFESTATION_OF", "rel3_METHOD_OF", "rel3_OCCURS_IN", "rel3_PART_OF", "rel3_PRECEDES", "rel3_PREDISPOSES", "rel3_PREVENTS", "rel3_PROCESS_OF", "rel3_PRODUCES", "rel3_same_as", "rel3_STIMULATES", "rel3_TREATS", "rel3_USES", "rel3_MENTIONED_IN", "rel3_HAS_MESH"]


X=data[feature_cols]
y=data["INTERACTS"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)

model = Sequential()
model.add(Dense(512, input_shape=(X_train.shape[1],)))
model.add(Activation('tanh'))
model.add(BatchNormalization())
                           
model.add(Dropout(0.1))   # Dropout helps protect the model from memorizing or "overfitting" the training data
model.add(Dense(128))
model.add(Activation('tanh'))

model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics=tf.keras.metrics.Recall())
model.fit(X_train, y_train,
          batch_size=128, epochs=100, verbose=2,
          validation_data=(X_test, y_test))

'''
new_df = pd.DataFrame(X_test).copy()
y_pred=model.predict(X_test)

new_df['preds'] = y_pred
new_df['labels'] = y_test
class_1_probs = model.predict(X_test)[:, 1]
new_df['proba']=class_1_probs
new_df.to_csv('KERASPROBAS.csv',index=False)
'''