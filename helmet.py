import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# Define image directories
image_dir = r'C:\Users\acer\Desktop\helmet detection\helmetdata\helmetdata'
no_helmet_images = os.listdir(image_dir + '/no_helmet')
helmet_images = os.listdir(image_dir + '/with_helmet')

print("--------------------------------------\n")
print('The length of No Helmet images:', len(no_helmet_images))
print('The length of Helmet images:', len(helmet_images))
print("--------------------------------------\n")

dataset = []
label = []
img_size = (128, 128)

# Process "No Helmet" images
for i, image_name in tqdm(enumerate(no_helmet_images), desc="Processing 'No Helmet' images"):
    if image_name.lower().endswith(('jpg', 'jpeg', 'png')):  # Handle common image formats
        try:
            image = cv2.imread(os.path.join(image_dir, 'no_helmet', image_name))
            if image is not None:
                image = Image.fromarray(image, 'RGB')
                image = image.resize(img_size)
                dataset.append(np.array(image))
                label.append(0)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

# Process "With Helmet" images
for i, image_name in tqdm(enumerate(helmet_images), desc="Processing 'With Helmet' images"):
    if image_name.lower().endswith(('jpg', 'jpeg', 'png')):  # Handle common image formats
        try:
            image = cv2.imread(os.path.join(image_dir, 'with_helmet', image_name))
            if image is not None:
                image = Image.fromarray(image, 'RGB')
                image = image.resize(img_size)
                dataset.append(np.array(image))
                label.append(1)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

dataset = np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length:', len(dataset))
print('Label Length:', len(label))
print("--------------------------------------\n")

if len(dataset) == 0 or len(label) == 0:
    print("Error: Dataset is empty. Check the image paths and file formats.")
    exit()

# Train-test split
print("--------------------------------------\n")
print("Train-Test Split")
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)
print("--------------------------------------\n")

# Normalize the dataset
print("--------------------------------------\n")
print("Normalizing the Dataset.\n")
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
print("--------------------------------------\n")

# Model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print("--------------------------------------\n")
model.summary()
print("--------------------------------------\n")

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
print("--------------------------------------\n")
print("Training Started.\n")
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
print("Training Finished.\n")
print("--------------------------------------\n")

# Save accuracy plot
plt.plot(history.epoch, history.history['accuracy'], label='accuracy')
plt.plot(history.epoch, history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig(r'C:\Users\acer\Desktop\helmet detection\helmet_sample_accuracy_plot.png')

# Clear the previous plot
plt.clf()

# Save loss plot
plt.plot(history.epoch, history.history['loss'], label='loss')
plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(r'C:\Users\acer\Desktop\helmet detection\helmet_sample_loss_plot.png')

# Evaluate the model
print("--------------------------------------\n")
print("Model Evaluation Phase.\n")
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")
y_pred=model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)
print('classification Report\n',classification_report(y_test,y_pred))
print("--------------------------------------\n")

model.save(r'C:\Users\acer\Desktop\helmet detection\cnn_helmet.h5')

print("--------------------------------------\n")
print("Model Prediction.\n")

# model = tf.keras.models.load_model('CNN/tumor_detection/results/model/cnn_tumor.h5')

def make_prediction(img,model):
    # img=cv2.imread(img)
    img=Image.fromarray(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)
    if res:
        print("Helmet detected")
    else:
        print("No Helmet detected")
    return res
        
make_prediction(cv2.imread(r'C:\Users\acer\Desktop\helmet detection\helmetdata\helmetdata\no_helmet\images-2022-12-19T201737-725_jpeg.rf.816d6ccfa4a781a6d0c7cd7b6bc1b9ae.jpg'),model)
print("--------------------------------------\n")
make_prediction(cv2.imread(r'C:\Users\acer\Desktop\helmet detection\helmetdata\helmetdata\with_helmet\13jan_RNJWSHI-W-MA14RAMNAD-HELMET__flip_jpg.rf.590466556897cc1334ecd6a4989f2621.jpg'),model)
print("--------------------------------------\n")
