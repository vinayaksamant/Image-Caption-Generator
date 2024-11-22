
import numpy as np
import os

image_dir = "/home/vinayak/image_caption_generator/fliker8k/images"
caption_file_path = "/home/vinayak/image_caption_generator/fliker8k/captions.txt"

def load_captions(filename):
    captions = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:  # Ensure that the line has at least two parts
                image_id, caption = parts[0].split('.')[0], parts[1]
                if image_id not in captions:
                    captions[image_id] = []
                captions[image_id].append(caption)
    return captions

captions = load_captions(caption_file_path)
print("captions loaded")

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def extract_image_features(model, image_path):
    img = preprocess_image(image_path)
    features = model.predict(img)
    return features

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet')
image_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Extract features for all images
image_features = {}
for image_id in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_id)
    image_features[image_id] = extract_image_features(image_model, image_path)

print('feature extracted')


vocab_size = 10000
max_length = 50
embedding_size = 256
lstm_units = 256

# LSTM Model
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(embedding_size, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_size, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(lstm_units)(se2)
decoder1 = tf.keras.layers.add([fe2, se3])
decoder2 = Dense(lstm_units, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.summary()

def data_generator(captions, image_features, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for key, desc_list in captions.items():
            n += 1
            photo = image_features[key][0]
            for desc in desc_list:
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == num_photos_per_batch:
                yield ([np.array(X1), np.array(X2)], np.array(y))
                X1, X2, y = [], [], []
                n = 0

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')
print("Training started")
# Train the model
epochs = 1
steps = len(captions)
wordtoix = {}
num_photos_per_batch = 32 
for i in range(epochs):
    generator = data_generator(captions, image_features, wordtoix, max_length, num_photos_per_batch)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('model_' + str(i) + '.h5')



