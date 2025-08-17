import cv2
import os
import numpy as np
import argparse
import sys
import requests
import mediapipe as mp
from openai import OpenAI

class EnhancedFaceDetector:
    def __init__(self, data_dir=None, models_dir=None):
        """
        Initialize the enhanced face detector with necessary cascade files and models
        
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
        
        # Initialize MediaPipe Face Mesh for detailed facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
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
        
        # Initialize virtual assistant
        self.assistant_enabled = False
        try:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key="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"
            )
            self.assistant_enabled = True
            print("Virtual assistant initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize virtual assistant: {e}")
        
        # Feedback messages based on emotions
        self.emotion_feedback = {
            "Happy": [
                "You seem happy today! That's great!",
                "Your smile brightens the room!",
                "Happiness looks good on you!",
                "Keep smiling, it suits you well!"
            ],
            "Neutral": [
                "How are you feeling today?",
                "Take a moment to breathe and relax.",
                "Remember to take breaks throughout your day.",
                "Sometimes a neutral face means deep thinking!"
            ]
        }
        
        # Predefined haircut recommendations by face shape and gender
        # Used as fallback when API is unavailable
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
            "heart": {
                "Male": "Styles with volume on the sides help balance your wider forehead. Try a side part or medium-length cut.",
                "Female": "Medium to long styles with layers starting at your chin can balance your heart-shaped face."
            },
            "oblong": {
                "Male": "Styles with volume on the sides and shorter length help balance your face. Try a textured crop or side part.",
                "Female": "Bobs, waves, or cuts with volume on the sides can make your face appear less long."
            },
            "diamond": {
                "Male": "Textured styles with some volume work well with your diamond face shape. Try a fringe or forward styling.",
                "Female": "Chin-length bobs, side-swept bangs, or styles with volume at the jawline complement your diamond face."
            },
            "unknown": {
                "Male": "A versatile medium-length cut with some texture would be a good choice for your face.",
                "Female": "A shoulder-length cut with layers would frame your face nicely and offer styling versatility."
            }
        }
            
        print("All cascade classifiers loaded successfully")
    
    def get_assistant_response(self, emotion, age_range, gender, head_pose=None, facial_structure=None):
        """
        Get a response from the virtual assistant based on detected attributes
        
        Args:
            emotion: Detected emotion (Happy/Neutral)
            age_range: Detected age range
            gender: Detected gender
            head_pose: Detected head pose (if available)
            facial_structure: Detected facial structure and proportions (if available)
            
        Returns:
            Assistant's response as string
        """
        if not self.assistant_enabled:
            return "Assistant not available"
        
        try:
            pose_info = ""
            if head_pose:
                pose_info = f" Their head is {head_pose}."
                
            facial_info = ""
            if facial_structure:
                facial_info = f" Their facial structure analysis: {facial_structure}."
                
            prompt = f"""The camera sees a {gender} person in the age range {age_range} who appears to be {emotion}.{pose_info}{facial_info}

Based on their facial structure analysis, what haircut or hairstyle would you recommend for them? 
Give a personalized recommendation that considers their facial proportions, gender, and age.
Keep your response brief (2-3 sentences) and conversational."""
            
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://face-detection-app.com",
                    "X-Title": "Face Detection App",
                },
                model="meta-llama/llama-3.3-70b-instruct:free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error getting assistant response: {e}")
            
            # Fallback to predefined recommendations if facial structure is available
            if facial_structure and "analysis" in facial_structure:
                face_shape = facial_structure["analysis"]["face_shape"]
                if face_shape in self.haircut_recommendations and gender in self.haircut_recommendations[face_shape]:
                    return self.haircut_recommendations[face_shape][gender]
            
            return "Based on your facial structure, I'd recommend a style that balances your proportions. Consider consulting with a professional stylist who can assess your features in person."
    
    def get_emotion_feedback(self, emotion):
        """
        Get feedback based on detected emotion
        
        Args:
            emotion: Detected emotion (Happy/Neutral)
            
        Returns:
            Feedback message as string
        """
        if emotion in self.emotion_feedback:
            return np.random.choice(self.emotion_feedback[emotion])
        return "How are you feeling today?"
    
    def analyze_facial_structure(self, image, landmarks):
        """
        Analyze facial structure based on facial landmarks
        
        Args:
            image: Input image
            landmarks: Facial landmarks from MediaPipe
            
        Returns:
            Dictionary containing facial structure analysis
        """
        if not landmarks:
            return None
            
        h, w, _ = image.shape
        
        # Extract key landmark points
        landmark_points = []
        for landmark in landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            landmark_points.append((x, y))
        
        # Calculate facial proportions
        # Face width at cheekbones (between points 123 and 352)
        face_width = np.linalg.norm(np.array(landmark_points[123]) - np.array(landmark_points[352]))
        
        # Face height (from chin to hairline, points 152 to 10)
        face_height = np.linalg.norm(np.array(landmark_points[152]) - np.array(landmark_points[10]))
        
        # Jaw width (between points 172 and 397)
        jaw_width = np.linalg.norm(np.array(landmark_points[172]) - np.array(landmark_points[397]))
        
        # Forehead width (between points 109 and 338)
        forehead_width = np.linalg.norm(np.array(landmark_points[109]) - np.array(landmark_points[338]))
        
        # Chin to cheekbones height (points 152 to 123/352 midpoint)
        cheekbone_point = (np.array(landmark_points[123]) + np.array(landmark_points[352])) / 2
        chin_to_cheekbones = np.linalg.norm(np.array(landmark_points[152]) - cheekbone_point)
        
        # Calculate ratios
        width_to_height_ratio = face_width / face_height if face_height > 0 else 0
        jaw_to_face_width_ratio = jaw_width / face_width if face_width > 0 else 0
        forehead_to_jaw_ratio = forehead_width / jaw_width if jaw_width > 0 else 0
        
        # Determine face shape based on proportions
        face_shape = "unknown"
        face_shape_confidence = 0
        
        # Oval face: width-to-height ratio around 0.75, balanced proportions
        if 0.65 <= width_to_height_ratio <= 0.85 and 0.7 <= jaw_to_face_width_ratio <= 0.9:
            face_shape = "oval"
            face_shape_confidence = 0.8
        
        # Round face: width-to-height ratio close to 1, full cheeks
        elif width_to_height_ratio > 0.85 and jaw_to_face_width_ratio > 0.9:
            face_shape = "round"
            face_shape_confidence = 0.8
        
        # Square face: width-to-height ratio around 0.85, strong jaw
        elif width_to_height_ratio > 0.8 and jaw_to_face_width_ratio > 0.9 and forehead_to_jaw_ratio < 1.1:
            face_shape = "square"
            face_shape_confidence = 0.8
        
        # Heart face: wider forehead, narrower jaw
        elif forehead_to_jaw_ratio > 1.2:
            face_shape = "heart"
            face_shape_confidence = 0.7
        
        # Long face: height much greater than width
        elif width_to_height_ratio < 0.65:
            face_shape = "oblong"
            face_shape_confidence = 0.7
        
        # Diamond face: narrow forehead and jawline, wide cheekbones
        elif jaw_to_face_width_ratio < 0.8 and forehead_to_jaw_ratio < 1.1:
            face_shape = "diamond"
            face_shape_confidence = 0.6
        
        # Create detailed analysis
        analysis = {
            "face_shape": face_shape,
            "confidence": face_shape_confidence,
            "proportions": {
                "width_to_height_ratio": round(width_to_height_ratio, 2),
                "jaw_to_face_width_ratio": round(jaw_to_face_width_ratio, 2),
                "forehead_to_jaw_ratio": round(forehead_to_jaw_ratio, 2)
            }
        }
        
        # Create a text description for the assistant
        description = f"face shape appears to be {face_shape} (confidence: {int(face_shape_confidence*100)}%), " + \
                      f"with width-to-height ratio of {round(width_to_height_ratio, 2)}, " + \
                      f"jaw-to-face width ratio of {round(jaw_to_face_width_ratio, 2)}, " + \
                      f"and forehead-to-jaw ratio of {round(forehead_to_jaw_ratio, 2)}"
        
        return {
            "analysis": analysis,
            "description": description
        }
    
    def estimate_head_pose(self, face_img, face_rect):
        """
        Estimate head pose using facial landmarks
        
        Args:
            face_img: Face region image
            face_rect: Face rectangle coordinates (x, y, w, h)
            
        Returns:
            Head pose description (looking left/right/straight)
        """
        x, y, w, h = face_rect
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate face center
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
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
        Detect faces in an image with age, gender, and emotion recognition
        
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
        
        # Process with MediaPipe for detailed facial landmarks
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
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
            
            # Detect smiles (for expression recognition)
            smiles = self.smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.7,
                minNeighbors=22,
                minSize=(25, 25)
            )
            
            # Determine emotion based on smile detection
            emotion = "Neutral"
            if len(smiles) > 0:
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 1)
                emotion = "Happy"
            
            # Estimate head pose
            head_pose = self.estimate_head_pose(img[y:y+h, x:x+w], (x, y, w, h))
            
            # Get facial structure analysis if MediaPipe landmarks are available
            facial_structure = None
            facial_structure_text = ""
            
            if results and results.multi_face_landmarks:
                # Analyze facial structure based on landmarks
                facial_structure = self.analyze_facial_structure(img, results.multi_face_landmarks[0])
                if facial_structure:
                    facial_structure_text = f"Face Shape: {facial_structure['analysis']['face_shape'].capitalize()}"
            
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
                
                # Get feedback based on emotion, age, and gender
                feedback = self.get_emotion_feedback(emotion)
                
                # Get assistant response if enabled
                if self.assistant_enabled:
                    facial_structure_desc = facial_structure["description"] if facial_structure else None
                    assistant_response = self.get_assistant_response(emotion, age, gender, head_pose, facial_structure_desc)
                    
                    # Display assistant response
                    cv2.putText(img, "Haircut Recommendation:", (10, img.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Split long responses into multiple lines
                    words = assistant_response.split()
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
                    
                    for i, line in enumerate(lines[:3]):  # Limit to 3 lines
                        cv2.putText(img, line, (10, img.shape[0] - 40 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            # Display emotion, age, gender, head pose, and facial structure
            cv2.putText(img, f"Emotion: {emotion}", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            if age_text:
                cv2.putText(img, age_text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            if gender_text:
                cv2.putText(img, gender_text, (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, f"Pose: {head_pose}", (x, y-80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            if facial_structure_text:
                cv2.putText(img, facial_structure_text, (x, y-100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw facial landmarks if available
        if results and results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw a subset of landmarks to avoid cluttering the image
                # Convert frozenset to list before slicing
                connections = list(self.mp_face_mesh.FACEMESH_TESSELATION)[:20]
                for connection in connections:  # Limit the number of connections drawn
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    start_point = face_landmarks.landmark[start_idx]
                    end_point = face_landmarks.landmark[end_idx]
                    
                    h, w, c = img.shape
                    start_x, start_y = int(start_point.x * w), int(start_point.y * h)
                    end_x, end_y = int(end_point.x * w), int(end_point.y * h)
                    
                    cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 1)
        
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
        
        # Initialize MediaPipe Face Mesh for video
        face_mesh_video = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect faces
            result_frame = self.detect_faces(frame)
            
            # Display the result
            cv2.imshow('Enhanced Face Detection', result_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        face_mesh_video.close()
        cap.release()
        cv2.destroyAllWindows()


def main():
    """
    Main function to parse arguments and run face detection
    """
    parser = argparse.ArgumentParser(description='Enhanced Face Detection using OpenCV')
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time detection')
    parser.add_argument('--output', help='Path to save output image (only for image mode)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode without GUI')
    parser.add_argument('--data-dir', help='Custom directory containing Haar cascade XML files')
    parser.add_argument('--models-dir', help='Custom directory containing age and gender models')
    parser.add_argument('--no-assistant', action='store_true', help='Disable virtual assistant integration')
    
    args = parser.parse_args()
    
    # Initialize face detector with optional custom directories
    detector = EnhancedFaceDetector(args.data_dir, args.models_dir)
    
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
            cv2.imshow('Enhanced Face Detection Result', result_img)
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
