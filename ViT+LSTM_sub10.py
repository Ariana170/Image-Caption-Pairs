import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dot, Dense, Dropout, Activation, Flatten, GRU, Concatenate
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Bidirectional, LSTM
from tensorflow.keras import metrics, Model
import numpy as np

tf.random.set_seed(1)
np.random.seed(1)

import cv2
import os
import pandas
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk.corpus import stopwords

#%%
#image preprocessing

#save the paths to the image folders
image_test_path = 'C:/Users/Asus user/PycharmProjects/pythonProject4/test_images'
image_train_path = 'C:/Users/Asus user/PycharmProjects/pythonProject4/train_images'
image_val_path = 'C:/Users/Asus user/PycharmProjects/pythonProject4/val_images'


def preprocess_image (path):
    image = cv2.imread(path) #read the images using the path
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert to RGB for later compatibility
    image_norm_tf = image_rgb / 255.0 #normalize from [0,255] to [0,1]

    final_tf = image_norm_tf.astype('float32') #change datatype to float32 for compatibility later on

    return final_tf

#%%
#text preprocessing

#save the csv files paths
test_csv_path = 'C:/Users/Asus user/PycharmProjects/pythonProject4/test.csv'
train_csv_path = 'C:/Users/Asus user/PycharmProjects/pythonProject4/train.csv'
val_csv_path = 'C:/Users/Asus user/PycharmProjects/pythonProject4/val.csv'

#read the csv files
test_csv = pandas.read_csv(test_csv_path)
train_csv = pandas.read_csv(train_csv_path)
val_csv = pandas.read_csv(val_csv_path)

def text_preprocessing(text):
    text = re.sub(r'[^a-zA-Z ]', '', text).lower() #remove all characters that are not letters
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))  #get all english stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words] #remove all stopwords
    lemm = nltk.WordNetLemmatizer()
    lemm_tokens = [lemm.lemmatize(filtered_token) for filtered_token in filtered_tokens] #apply lemmantizetion
    clean_text = ' '.join(lemm_tokens) #join words to reform the caption
    return clean_text

#%%
#save the clean text in the csv for easier use later on
test_csv['clean_text'] = [text_preprocessing(sentence) for sentence in test_csv['caption']]
train_csv['clean_text'] = [text_preprocessing(sentence) for sentence in train_csv['caption']]
val_csv['clean_text'] = [text_preprocessing(sentence) for sentence in val_csv['caption']]

#delete the caption from the csv copies
test_csv = test_csv.drop(['caption'], axis=1)
train_csv = train_csv.drop(['caption'], axis=1)
val_csv = val_csv.drop(['caption'], axis=1)

#%%

#Function which goes through the csv files row by row and gets the clean text, labels,
#ids and gets the image path that corresponds to each row
def get_image_paths_clean_captions_id(image_csv,image_path_dir):
    image_paths = []
    clean_captions = []
    labels = []
    ids = []
    for _, row in image_csv.iterrows():
        image_path = image_path_dir + '/' + str(row['image_id']) + '.jpg' #get the image path
        if os.path.isfile(image_path): #is image path is correct
            image_paths.append(image_path) #append the image path
            clean_captions.append(row['clean_text']) #append the clean caption
            ids.append(row['id']) #append the id
            if image_path_dir != image_test_path: #if it's not the test csv
                labels.append(row['label']) #append the label
        else:
            print(f"Skipping: {row['image_id']}") #print the image that was skipped if it didn't iin the directory
    return image_paths, clean_captions, labels, ids

image_train_paths, train_clean_captions, y_train, id_train = get_image_paths_clean_captions_id(train_csv, image_train_path)
image_test_paths, test_clean_captions, y_test, id_test = get_image_paths_clean_captions_id(test_csv, image_test_path)
image_val_paths, val_clean_captions, y_val, id_val = get_image_paths_clean_captions_id(val_csv, image_val_path)

#%%
#Preparing the text arrays for the model

#Instantitate a Tokenizer() object
tokenizer = Tokenizer()

#Fit that object on the cleaned training text
tokenizer.fit_on_texts(train_clean_captions)

#Call texts_to_sequences() for the training, the validation and the test text
train_seq = tokenizer.texts_to_sequences(train_clean_captions)
val_seq = tokenizer.texts_to_sequences(val_clean_captions)
test_seq = tokenizer.texts_to_sequences(test_clean_captions)

#Pad sequences
maxlen = max(len(seq) for seq in train_seq) #get the maximum length for the clean captions
x_train_text = np.int32(sequence.pad_sequences(train_seq, maxlen=maxlen, padding='post', dtype='int32'))
x_val_text = np.int32(sequence.pad_sequences(val_seq, maxlen=maxlen, padding='post', dtype='int32'))
x_test_text = np.int32(sequence.pad_sequences(test_seq, maxlen=maxlen, padding='post', dtype='int32'))

#%%
#Preparing the image arrays for the model

x_train_image = np.float32([preprocess_image(path) for path in image_train_paths])
x_val_image = np.float32([preprocess_image(path) for path in image_val_paths])
x_test_image = np.float32([preprocess_image(path) for path in image_test_paths])

#%%
#Image Model

class ImagePatches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches( #Extract patches
            images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dim = patches.shape[-1]
        patches = tf.reshape(patches, [tf.shape(images)[0], -1, patch_dim]) #Flatten patches
        return patches

def get_vision_transformer():
    image_input = Input(shape=(100, 100, 3), dtype='float32') #Input layer

    patches = ImagePatches(25)(image_input) #Extract the patches and flatten them
    linear_projection = Dense(64)(patches) #Linear Projection Layer
    position_embedding = Embedding(16,64)(tf.range(0, 16,1)) #Positional embedding

    tokens = linear_projection + position_embedding #Patch and position embedding sum

    #Transformer encoder:
    for i in range(5):
        tokens_norm = LayerNormalization()(tokens) #Normalization
        attention_head = MultiHeadAttention(3, 64)(tokens_norm, tokens_norm) # Multi-Head attention layer
        attention_head = Dropout(0.3)(attention_head) #Dropout layer to avoid overfitting

        tokens = tokens + attention_head

        tokens_1_norm = LayerNormalization()(tokens) #Normalization
        mlp_output = Dense(256, activation='relu')(tokens_1_norm) #Fully connected layer
        mlp_output = Dropout(0.4)(mlp_output)#Dropout layer to avoid overfitting
        mlp_output = Dense(64, activation='relu')(mlp_output) #Fully connected layer

        tokens = tokens + mlp_output

    tokens_final = LayerNormalization()(tokens) #Normalization
    tokens_final = Flatten()(tokens_final)
    tokens_final = Dropout(0.3)(tokens_final) #Dropout layer to avoid overfitting
    output = Dense(512, activation='relu')(tokens_final)#Fully connected layer
    output = Dropout(0.5)(output) #Dropout layer to avoid overfitting
    output = Dense(128, activation='relu')(output)#Fully connected layer

    model = Model(inputs=image_input, outputs=output) #Create the model
    model.summary() #View the model summary
    return model

#image_model = get_image_encoder()

#%%
#Text Model

max_features = len(tokenizer.word_index) + 1 #Get the vocabulary size

def get_text_encoder():
    model = Sequential() #Instantiate the model
    model.add(Embedding(max_features, 128, input_length=maxlen)) #Add an embedding layer
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))#Add a dense layer for the output,
                                                                            # the same size as the one in the text model
    #model.summary() #to view the model summary
    return model


#%%
#Matching Model
from tensorflow.keras.regularizers import l2

def match_model():
    image_input = Input(shape=(100,100,3), name = 'image_input')
    text_input = Input(shape=(None,), name = 'text_input')

    image_embedding = get_image_encoder()(image_input) #call the image encoder
    text_embedding = get_text_encoder()(text_input) #call the text encoder

    similarity = Dot(axes=-1, normalize=True)([image_embedding, text_embedding]) #Calculate the cosine similarity
    output = Dense(1, activation='sigmoid')(similarity) #Map the cosine similarity output from [-1,1] to [0,1]
                                                              #using the sigmoid activation function

    model = Model(inputs=[image_input, text_input], outputs=output, name='MatchModel')

    return  model

#%%
#Compile the model

model = match_model() #Call the match model

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, clipnorm=1.2) #get the optimizer

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])

#%%
#Implement early stopping to avoid overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)
#%%
#Reshape the labels
y_train = np.array(y_train).reshape(10000,1)
y_val = np.array(y_val).reshape(3000,1)

#%%
# Train the model

history = model.fit(
    x=[x_train_image, x_train_text],  # Pass both inputs
    y=y_train,
    epochs=15,
    batch_size=64,
    validation_data=([x_val_image, x_val_text], y_val), #use the validation data
    callbacks=[early_stopping, lr_schedule]
)

#%%
#Model Evaluation

loss, accuracy, precision, recall = model.evaluate([x_val_image, x_val_text], y_val)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

#%%
#Test Step
# Prediction for the validation data
predictions_val = model.predict([x_val_image, x_val_text])
# predicted label
y_val_pred = (predictions_val > 0.5).astype(int).flatten()

#%%
#Create the Confusion Matrix for the validation data
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(y_val, y_val_pred, normalize="pred") #create the confusion matrix

confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix) #create the display

confusion_matrix_display.plot() #plot the display
plt.show() #show the plot

#%%
#Create the accuracy and loss plots for the validation and train data (code from the second lab solution)

# plot loss
plt.subplot(211)
plt.title('Cross Entropy Loss')
plt.plot(history.history['loss'], color='blue', label='train')
plt.plot(history.history['val_loss'], color='orange', label='test')

# plot accuracy
plt.subplot(212)
plt.title('Classification Accuracy')
plt.plot(history.history['acc'], color='blue', label='train')
plt.plot(history.history['val_acc'], color='orange', label='test')

plt.show()

#%%
#Test Step
# Prediction
predictions = model.predict([x_test_image, x_test_text])
# predicted label
pred_labels = (predictions > 0.5).astype(int).flatten()

#%%
#Creating the submission CSV

submission_df = pandas.DataFrame(columns=['id', 'label'])
submission_df['label'] = pred_labels
submission_df['id'] = id_test
submission_df.to_csv('submission_name.csv', index=False)


