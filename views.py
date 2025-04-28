def udashboard(request):
    return render(request, 'udashboard.html')


def prediction(request):
    return render(request, 'prediction.html')


from django.shortcuts import render, redirect
from django.contrib import messages
from .models import User  # Assuming this is a custom user model; change if using Django's built-in User
from django.core.files.storage import FileSystemStorage


def register(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        mobile = request.POST.get('mobile')
        email = request.POST.get('email')
        password = request.POST.get('password')
        age = request.POST.get('age')
        address = request.POST.get('address')

        profile_picture = request.FILES.get('profile_picture')  # Handle file upload

        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered')
            return redirect('register')

        user = User(name=name, mobile=mobile, email=email, password=password, age=age, address=address)

        if profile_picture:
            fs = FileSystemStorage()
            filename = fs.save(profile_picture.name, profile_picture)
            user.profile_picture = filename

        user.save()

        messages.success(request, 'Registration successful! Please login.')
        return redirect('userlogin')

    return render(request, 'register.html')



def userlogin(request):
    if request.method == 'POST':
        email = request.POST.get('email')  # Get the username or email
        password = request.POST.get('password')  # Get the password

        # Check if the user exists and the password is correct
        try:
            user = User.objects.get(email=email)
            if user.password == password:  # Be cautious about plain text password comparison
                # Log the user in (you may want to set a session or token here)
                request.session['user_id'] = user.id  # Store user ID in session
                messages.success(request, 'Login successful!')
                return redirect('udashboard')  # Redirect to the index page or desired page
            else:
                messages.error(request, 'Invalid email or password. Please try again.')
        except User.DoesNotExist:
            messages.error(request, 'Invalid email or password. Please try again.')

    return render(request, 'userlogin.html')
def profile(req):
    user_id = req.session.get("user_id")  # Use 'get' to avoid a crash if session is not set
    user = User.objects.get(id=user_id)  # Fetch the user by their id
    
    if req.method == 'POST':
        user_name = req.POST.get('userName')
        user_age = req.POST.get('userAge')
        user_phone = req.POST.get('userPhNum')
        user_email = req.POST.get('userEmail')
        user_address = req.POST.get("userAddress")

        # Update user details
        user.name = user_name
        user.age = user_age
        user.address = user_address
        user.mobile = user_phone
        user.email = user_email

        # Handle profile picture update
        if len(req.FILES) != 0:
            image = req.FILES['profilepic']
            user.profile_picture = image  # Save the new profile image

        user.save()  # Save the user details after updating all fields
        messages.success(req, 'Profile updated successfully!')

    context = {"i": user}  # 'i' contains user details to be used in the template
    return render(req, 'profile.html', context)




import os
import numpy as np
import cv2
from django.shortcuts import render
from tensorflow.keras.models import load_model
from django.core.files.storage import FileSystemStorage
from skimage.measure import shannon_entropy

# Load the trained classification model
model = load_model('vgg16_model.h5')  # Ensure correct model path

# Function to check if an image is grayscale
def is_ultrasound_image(image_path, entropy_threshold=5.0):
    """Detects if an image is grayscale and has expected entropy."""
    img = cv2.imread(image_path)

    if img is None:
        return False  # Invalid image

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute entropy
    entropy = shannon_entropy(gray_img)

    print(f"Image Entropy: {entropy}")  # Debugging output

    # Ultrasound images usually have entropy between ~4.0 and 7.0
    return entropy_threshold <= entropy <= 7.5  

# Preprocessing function for classification
def preprocess_for_classification(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Invalid image file. Please upload a valid ultrasound image.")

    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0  # Normalize pixel values
    img_rgb = np.stack([img, img, img], axis=-1)  # Convert grayscale to RGB
    return np.expand_dims(img_rgb, axis=0)

# Function to predict image class
def predict_classification(image_path):
    if not is_ultrasound_image(image_path):
        raise ValueError("Invalid image. Please upload a valid ultrasound scan.")

    img = preprocess_for_classification(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class[0]

# Function to handle file upload and prediction
def prediction(request):
    prediction_result = None
    error_message = None

    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']

        # Save uploaded image
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        image_path = os.path.join(fs.location, filename)

        try:
            # Perform validation and prediction
            predicted_class = predict_classification(image_path)
            prediction_result = 'Intact' if predicted_class == 0 else 'Torn'

        except ValueError as e:
            error_message = str(e)

        # Remove uploaded image
        if os.path.exists(image_path):
            os.remove(image_path)

    return render(request, 'prediction.html', {'prediction': prediction_result, 'error': error_message})
