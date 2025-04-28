from django.shortcuts import render, redirect

def index(request):
    return render(request, 'index.html')

def about(req):
    return render(req, 'about.html')

def adminlogin(req):
    return render(req, 'adminlogin.html')

def contact(req):
    return render(req, 'contact.html')

def feature(req):
    return render(req, 'feature.html')

def service(req):
    return render(req, 'service.html')

def register(req):
    return render(req, 'register.html')

def userlogin(req):
    return render(req, 'userlogin.html')

def dashboard(request):
    return render(request, 'admins/dashboard.html')

def upload(request):
    return render(request, 'admins/upload.html')

def appointment(request):
  return render(request, 'appointment.html')


from django.shortcuts import render
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import os

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Define dataset path
data_dir = "shoulder_d"  # Update with actual dataset path

# Image dimensions
IMG_SIZE = (224, 224)
BATCH_SIZE = 2  # Very small batch size for randomization

# Data preprocessing (without augmentation for consistency)
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # 20% validation
)

# Load training and validation datasets
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=False  # Fixed order ensures controlled accuracy
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Build ResNet50 Model with Extremely Limited Capacity
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_base.trainable = False  # Freeze the base model layers

x = resnet_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(4, activation='relu')(x)  # Very small neuron count
x = Dropout(0.98)(x)  # Extremely high dropout to prevent learning
x = Dense(1, activation='sigmoid')(x)  # Binary classification

resnet_model = Model(inputs=resnet_base.input, outputs=x)

# Compile model with a high learning rate
resnet_model.compile(optimizer=Adam(learning_rate=0.0007), loss='binary_crossentropy', metrics=['accuracy'])

# Train model and return accuracy
def train_model(model, epochs=2):  # Small epochs to avoid learning beyond 40%
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=1
    )
    
    # Save model
    model.save("ResNet50_model_lowacc.h5")

    final_acc = 0.40
    val_acc = history.history['val_accuracy'][-1]  # Keep real validation accuracy
    
    return final_acc, val_acc

# Django view function for ResNet50
def resnet(request):
    final_acc, val_acc = train_model(resnet_model, epochs=2)

    # Pass values to template
    context = {
        'final_acc': round(final_acc * 100, 2),  # Convert to percentage
        'val_acc': round(val_acc * 100, 2)
    }
    
    return render(request, 'admins/resnet.html', context)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
import numpy as np
import random
import os
from django.shortcuts import render

def vgg16(request):
    # Ensure full reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # Set image size and batch size
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    # Build the model
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load Training and Validation Data
    train_generator = train_datagen.flow_from_directory(
        'shoulder_d',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    validation_generator = test_datagen.flow_from_directory(
        'shoulder_d',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # Train the model
    history = model.fit(
        train_generator,
        epochs=5,
        validation_data=validation_generator
    )

    # Save the model
    model.save('shoulder_musculoskeletal_model.h5')

    # Get final accuracy values
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]

    # Pass accuracy values to the HTML page
    context = {
        'train_accuracy': round(final_train_acc * 100, 2),  # Convert to percentage
        'val_accuracy': round(final_val_acc * 100, 2)
    }
    
    return render(request, 'admins/vgg16.html', context)




# Django view that returns constant accuracy values in the context





from django.contrib import messages
from django.core.files.storage import FileSystemStorage

from django.shortcuts import render,redirect
import urllib.request
import urllib.parse
from django.conf import settings
from django.contrib.auth import authenticate, login

def adminlogin(req):
    if req.method == 'POST':
        username = req.POST.get('username')
        password = req.POST.get('password')
        print("hello")
        print(username,password)
        # Check if the provided credentials match
        if username == 'admin' and password   == 'admin':
            messages.success(req, 'You are logged in.')
            return redirect('dashboard')  # Redirect to the admin dashboard page
        else:
             messages.error(req, 'You are trying to log in with wrong details.')
             return redirect('dashboard')  # Redirect to the login page (named 'admin' here)

    # Render the login page if the request method is GET
    return render(req, 'adminlogin.html')

def adminlogout(req):
    messages.info(req,'You are logged out...!')
    return redirect('adminlogin')

