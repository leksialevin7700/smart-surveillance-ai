import cv2
import smtplib
import os  # For accessing environment variables
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage  # For attaching images to emails
from dotenv import load_dotenv
import datetime  # Import datetime for logging
import time  # Import time module

load_dotenv() 

def send_email(subject, body, image_path=None):
    sender_email = os.getenv("SENDER_EMAIL")  
    receiver_email = os.getenv("RECEIVER_EMAIL")  
    password = os.getenv("EMAIL_PASSWORD") 

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach the image if it exists
    if image_path:
        with open(image_path, 'rb') as img:
            img_data = MIMEImage(img.read(), name=os.path.basename(image_path))
            msg.attach(img_data)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

def log_motion_event():
    with open("motion_log.txt", "a") as log_file:
        log_file.write(f"Motion detected at {datetime.datetime.now()}\n")

def capture_frame(frame):
    # Create a filename based on the current timestamp
    filename = f"motion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    # Save the frame as an image
    cv2.imwrite(filename, frame)
    print(f"Captured and saved frame: {filename}")
    return filename  # Return the filename for email attachment

# Initialize video capture with video file path
cap = cv2.VideoCapture('vtest.avi')  # or use the path where you saved the video

# Read the first two frames
_, frame1 = cap.read()
_, frame2 = cap.read()

motion_detected = False  # Flag to track if motion was detected
last_alert_time = 0  # To track when the last alert was sent
alert_cooldown = 10  # Cooldown period in seconds
camera_alert_sent = False  # Flag to track if camera alert has been sent

while True:  # Run indefinitely until manually stopped
    # Check if camera feed is open
    if not cap.isOpened():
        if not camera_alert_sent:  # Send alert only if not already sent
            send_email("Camera Error", "Camera feed has been lost or disconnected.")
            camera_alert_sent = True  # Set the flag to avoid duplicate alerts
        continue  # Skip the rest of the loop if the camera feed is lost

    # Calculate the difference between frames
    diff = cv2.absdiff(frame1, frame2)

    # Convert to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # Dilate the threshold image to fill in holes
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Only consider large contours
        if cv2.contourArea(contour) > 5000:
            # Draw rectangle around the detected motion
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

            current_time = time.time()  # Get the current time
            # Check if the cooldown period has passed
            if current_time - last_alert_time > alert_cooldown:
                # Capture the current frame
                image_path = capture_frame(frame1)  # Capture and save the frame

                send_email("Motion Detected", "ALERT ALERT!!!! Human motion has been detected in the monitored area.", image_path)
                log_motion_event()  # Log the motion event
                last_alert_time = current_time  # Update the last alert time
                motion_detected = True  # Set flag to true to prevent multiple emails

    if motion_detected and not any(cv2.contourArea(contour) > 5000 for contour in contours):
        motion_detected = False  # Reset flag if no large contours are detected

    # Display the frames
    cv2.imshow("Motion Detection", frame1)

    # Update frames
    frame1 = frame2
    _, frame2 = cap.read()

    if cv2.waitKey(10) == 27:  # Escape key to exit
        break

cap.release()
cv2.destroyAllWindows()
