Automated Detection of Shoulder Rotator Cuff Tendon Tears
A Deep Learning Full Stack Application using VGG16 and ResNet50 with Ultrasound Imaging

#Overview
DeepTear is a full-stack web application for the automated detection of shoulder rotator cuff tendon tears using ultrasound images. This project combines Django, deep learning (VGG16 and ResNet50 models), and a modern web interface to assist doctors and radiologists by providing quick, AI-driven diagnoses.

The system allows users to upload ultrasound scans, receive model predictions, visualize results, and download diagnostic reports through a clean, secure web platform.

#Features
Upload ultrasound images for real-time analysis.

Classification using pre-trained deep learning models (VGG16 and ResNet50).

User authentication and admin panel.

Detailed diagnostic report generation.

Admin dashboard to manage users, images, and reports.

Scalable and cloud deployment ready (AWS, Heroku, etc.).

#Tech Stack
Frontend: HTML5, CSS3, JavaScript, Bootstrap

Backend: Django (Python 3)

Deep Learning Models: TensorFlow/Keras (VGG16, ResNet50)

Database: SQLite / PostgreSQL

Deep Learning Models
VGG16: Fine-tuned for tendon tear detection, known for deep but simple architecture.

ResNet50: Residual network capable of handling deeper and more complex feature extractions.

Both models were trained and validated on an ultrasound dataset of rotator cuff images.

