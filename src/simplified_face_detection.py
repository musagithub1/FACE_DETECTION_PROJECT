import cv2
import os
import numpy as np
import argparse
import sys
import requests
import random

class SimplifiedFaceDetector:
    def __init__(self, data_dir=None, models_dir=None):
        """
        Initialize the simplified face detector with necessary cascade files and models
        
        Args:
            data_dir: Directory containing the Haar cascade XML files. If None, uses relative path.
            models_dir: Directory containing age and gender models. If None, uses relative path.
        """
        # If no data directory is specified, use a relative path approach
        if data_dir is None:
            # Get the directory where the script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to the project root
            project_root = os.path.dirname(script_dir)
            # Set data directory relative to project root
            data_dir = os.path.join(project_root, 'data')
        
        # If no models directory is specified, use a relative path approach
        if models_dir is None:
            # Get the directory where the script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to the project root
            project_root = os.path.dirname(script_dir)
            # Set models directory relative to project root
            models_dir = os.path.join(project_root, 'models')
        
        self.data_dir = data_dir
        self.models_dir = models_dir
        
        print(f"Looking for cascade files in: {data_dir}")
        print(f"Looking for models in: {models_dir}")
        
        # Load the face cascade
        face_cascade_path = os.path.join(data_dir, 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Load the eye cascade
        eye_cascade_path = os.path.join(data_dir, 'haarcascade_eye.xml')
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        # Load the smile cascade for expression detection
        smile_cascade_path = os.path.join(data_dir, 'haarcascade_smile.xml')
        self.smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
        
        # Check if cascades loaded successfully
        if self.face_cascade.empty():
            raise ValueError(f"Error loading face cascade from {face_cascade_path}")
        if self.eye_cascade.empty():
            raise ValueError(f"Error loading eye cascade from {eye_cascade_path}")
        if self.smile_cascade.empty():
            raise ValueError(f"Error loading smile cascade from {smile_cascade_path}")
        
        # Load age and gender models
        self.age_net = None
        self.gender_net = None
        
        try:
            # Load age model
            age_proto = os.path.join(models_dir, 'age_deploy.prototxt')
            age_model = os.path.join(models_dir, 'age_net.caffemodel')
            self.age_net = cv2.dnn.readNet(age_proto, age_model)
            
            # Load gender model
            gender_proto = os.path.join(models_dir, 'gender_deploy.prototxt')
            gender_model = os.path.join(models_dir, 'gender_net.caffemodel')
            self.gender_net = cv2.dnn.readNet(gender_proto, gender_model)
            
            # Define age and gender labels
            self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
            self.gender_list = ['Male', 'Female']
            
            print("Age and gender models loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load age and gender models: {e}")
            self.age_net = None
            self.gender_net = None
        
        # Feedback messages based on emotions
        self.emotion_feedback = {
            "happy": [
                "You seem happy today! That's great!",
                "Your smile brightens the room!",
                "Happiness looks good on you!",
                "Keep smiling, it suits you well!"
            ],
            "neutral": [
                "How are you feeling today?",
                "Take a moment to breathe and relax.",
                "Remember to take breaks throughout your day.",
                "Sometimes a neutral face means deep thinking!"
            ]
        }
        
        # Predefined haircut recommendations by face shape and gender
        self.haircut_recommendations = {
            "oval": {
                "Male": "Most hairstyles work well with your oval face shape. Try a classic short back and sides with some length on top.",
                "Female": "Your oval face shape suits most hairstyles. Consider layers that frame your face or a medium-length cut."
            },
            "round": {
                "Male": "Styles with height on top and shorter sides help elongate your round face. Try a pompadour or textured quiff.",
                "Female": "Long layers past the chin or an asymmetrical bob can help elongate your round face shape."
            },
            "square": {
                "Male": "Your strong jawline works well with short sides and textured top. Consider a crew cut or textured crop.",
                "Female": "Soft layers, side-swept bangs, or a lob (long bob) can soften your square jawline."
            },
            "unknown": {
                "Male": "A versatile medium-length cut with some texture would be a good choice for your face.",
                "Female": "A shoulder-length cut with layers would frame your face nicely and offer styling versatility."
            }
        }
            
        print("All cascade classifiers loaded successfully")
    
    def get_emotion_feedback(self, emotion):
        """
        Get feedback based on detected emotion
        
        Args:
            emotion: Detected emotion (happy, neutral)
            
        Returns:
            Feedback message as string
        """
        emotion = emotion.lower()
        if emotion in self.emotion_feedback:
            return random.choice(self.emotion_feedback[emotion])
        return "How are you feeling today?"
    
    def get_haircut_recommendation(self, face_shape, gender):
        """
        Get haircut recommendation based on face shape and gender
        
        Args:
            face_shape: Detected face shape
            gender: Detected gender
            
        Returns:
            Haircut recommendation as string
        """
        if face_shape in self.haircut_recommendations and gender in self.haircut_recommendations[face_shape]:
            return self.haircut_recommendations[face_shape][gender]
        return self.haircut_recommendations["unknown"][gender]
    
    def estimate_face_shape(self, face_img):
        """
        Estimate face shape based on simple proportions
        
        Args:
            face_img: Face region image
            
        Returns:
            Estimated face shape as string
        """
        h, w = face_img.shape[:2]
        
        # Calculate face width to height ratio
        ratio = w / h if h > 0 else 0
        
        # Simple face shape estimation based on width-to-height ratio
        if 0.85 <= ratio <= 0.95:
            return "oval"
        elif ratio > 0.95:
            return "round"
        elif ratio < 0.85:
            return "square"
        else:
            return "unknown"
    
    def estimate_head_pose(self, face_img, face_rect):
        """
        Estimate head pose using eye positions
        
        Args:
            face_img: Face region image
            face_rect: Face rectangle coordinates (x, y, w, h)
            
        Returns:
            Head pose description (looking left/right/straight)
        """
        x, y, w, h = face_rect
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes in the face region
        eyes = self.eye_cascade.detectMultiScale(gray)
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            
            # Calculate eye centers
            eye_centers = []
            for (ex, ey, ew, eh) in eyes[:2]:  # Take only the first two eyes
                eye_center_x = ex + ew // 2
                eye_center_y = ey + eh // 2
                eye_centers.append((eye_center_x, eye_center_y))
            
            # Calculate angle between eyes
            if len(eye_centers) == 2:
                left_eye, right_eye = eye_centers
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                
                # Calculate angle in degrees
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Determine head pose based on angle
                if angle < -10:
                    return "tilted right"
                elif angle > 10:
                    return "tilted left"
                else:
                    # Check horizontal position of eyes relative to face center
                    eye_x_avg = (left_eye[0] + right_eye[0]) / 2
                    face_width_ratio = (eye_x_avg - x) / w
                    
                    if face_width_ratio < 0.45:
                        return "looking right"
                    elif face_width_ratio > 0.55:
                        return "looking left"
                    else:
                        return "looking straight"
        
        return "position unknown"
    
    def detect_faces(self, img):
        """
        Detect faces in an image with age, gender, and basic emotion recognition
        
        Args:
            img: Input image
            
        Returns:
            Processed image with face detection visualization
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw rectangles around faces and detect features
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Region of interest for face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Estimate face shape
            face_shape = self.estimate_face_shape(img[y:y+h, x:x+w])
            face_shape_text = f"Face Shape: {face_shape.capitalize()}"
            
            # Estimate head pose
            head_pose = self.estimate_head_pose(img[y:y+h, x:x+w], (x, y, w, h))
            
            # Basic emotion detection using smile cascade
            smiles = self.smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.7,
                minNeighbors=22,
                minSize=(25, 25)
            )
            
            # Determine emotion based on smile detection
            emotion = "neutral"
            if len(smiles) > 0:
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 1)
                emotion = "happy"
            
            emotion_text = f"Emotion: {emotion.capitalize()}"
            emotion_feedback = self.get_emotion_feedback(emotion)
            
            # Get age and gender if models are available
            age_text = ""
            gender_text = ""
            age = ""
            gender = ""
            
            if self.age_net is not None and self.gender_net is not None:
                # Create a face blob
                face_img = img[y:y+h, x:x+w].copy()
                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                
                # Predict gender
                self.gender_net.setInput(blob)
                gender_preds = self.gender_net.forward()
                gender = self.gender_list[gender_preds[0].argmax()]
                gender_text = f"Gender: {gender}"
                
                # Predict age
                self.age_net.setInput(blob)
                age_preds = self.age_net.forward()
                age = self.age_list[age_preds[0].argmax()]
                age_text = f"Age: {age}"
                
                # Get haircut recommendation
                haircut_recommendation = self.get_haircut_recommendation(face_shape, gender)
                
                # Display haircut recommendation
                cv2.putText(img, "Recommendation:", (10, img.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Split long recommendation into multiple lines
                words = haircut_recommendation.split()
                lines = []
                current_line = ""
                
                for word in words:
                    if len(current_line + " " + word) <= 50:  # Limit line length
                        current_line += " " + word if current_line else word
                    else:
                        lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                
                for i, line in enumerate(lines[:4]):  # Limit to 4 lines
                    cv2.putText(img, line, (10, img.shape[0] - 60 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            # Display emotion, age, gender, head pose, and facial structure
            cv2.putText(img, emotion_text, (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            if age_text:
                cv2.putText(img, age_text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            if gender_text:
                cv2.putText(img, gender_text, (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, f"Pose: {head_pose}", (x, y-80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, face_shape_text, (x, y-100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Display emotion feedback in the top-left corner
            if emotion_feedback:
                cv2.putText(img, f"Feedback: {emotion_feedback}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Display count of faces
        cv2.putText(img, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return img
    
    def process_image(self, image_path):
        """
        Process a single image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Processed image
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Detect faces
        result_img = self.detect_faces(img)
        
        return result_img
    
    def process_webcam(self):
        """
        Process video from webcam with real-time face detection
        """
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        print("Webcam initialized. Press 'q' to quit.")
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect faces
            result_frame = self.detect_faces(frame)
            
            # Display the result
            cv2.imshow('Simplified Face Detection', result_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


def main():
    """
    Main function to parse arguments and run face detection
    """
    parser = argparse.ArgumentParser(description='Simplified Face Detection using OpenCV')
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time detection')
    parser.add_argument('--output', help='Path to save output image (only for image mode)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode without GUI')
    parser.add_argument('--data-dir', help='Custom directory containing Haar cascade XML files')
    parser.add_argument('--models-dir', help='Custom directory containing age and gender models')
    
    args = parser.parse_args()
    
    # Initialize face detector with optional custom directories
    detector = SimplifiedFaceDetector(args.data_dir, args.models_dir)
    
    if args.webcam:
        if args.headless:
            print("Webcam mode cannot be used in headless mode")
            return
        # Process webcam feed
        detector.process_webcam()
    elif args.image:
        # Process single image
        result_img = detector.process_image(args.image)
        
        # Display the result if not in headless mode
        if not args.headless:
            cv2.imshow('Simplified Face Detection Result', result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Save output if specified
        if args.output:
            cv2.imwrite(args.output, result_img)
            print(f"Result saved to {args.output}")
    else:
        print("Error: Please specify either --image or --webcam")
        parser.print_help()


if __name__ == "__main__":
    main()
