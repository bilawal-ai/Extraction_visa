import streamlit as st
import cv2
import pytesseract
from ultralytics import YOLO
import os
from PIL import Image
from datetime import datetime
import numpy as np

# Set up Tesseract executable path if required (uncomment and set the correct path)
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Load the YOLO model
model_path = 'best.pt'  # Make sure to set the correct path to your YOLO model
model = YOLO(model_path)

# Streamlit app title
st.title("Visa Information Extractor")

# Camera input
camera_image = st.camera_input("Take a picture of the visa using your camera")

# If an image is captured via the camera
if camera_image is not None:
    # Load the image using PIL
    image = Image.open(camera_image)
    
    # Display the captured image
    st.image(image, caption="Captured Visa Image", use_column_width=True)
    
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Save the captured image for YOLO model to process (if necessary)
    image_path = "captured_visa.jpg"
    image.save(image_path)
    
    # Run inference with YOLO model
    results = model.predict(image_path)
    
    # Dictionary to hold the OCR results with labels
    ocr_results = {}

    # Iterate over each result (assuming single image inference)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        labels = result.names  # Detected class names
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

        # Iterate over each detected object (bounding box)
        for i, (box, label, confidence) in enumerate(zip(boxes, labels, confidences)):
            x1, y1, x2, y2 = map(int, box)  # Get the coordinates of the bounding box
            cropped_region = image_cv[y1:y2, x1:x2]  # Crop the image to the bounding box region

            # Convert cropped region to RGB (from BGR)
            cropped_region_rgb = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB)
            cropped_image_pil = Image.fromarray(cropped_region_rgb)

            # Now apply OCR to the cropped region
            extracted_text = pytesseract.image_to_string(cropped_image_pil, config='--psm 6').strip()
            extracted_text = ' '.join(extracted_text.replace('\n', ' ').split())  # Normalize spaces and remove new lines
            extracted_text = extracted_text.strip('()\'\"-`,.:;|!?')  # Strip unwanted punctuations

            # Append or set text in dictionary to prevent overwriting
            if label in ocr_results:
                ocr_results[label] += ' ' + extracted_text
            else:
                ocr_results[label] = extracted_text

    # Process OCR results and extract information
    dates = []
    passport_number = None
    full_name = None

    for label, text in ocr_results.items():
        date_parsed = False
        for date_format in ['%Y/%m/%d', '%d/%m/%Y']:
            try:
                parsed_date = datetime.strptime(text, date_format)
                dates.append((label, parsed_date))
                date_parsed = True
                break
            except ValueError:
                continue

        if not date_parsed:
            if text.isdigit() or (any(char.isdigit() for char in text) and any(char.isalpha() for char in text)):
                passport_number = text
            else:
                full_name = text

    # Sort dates to find the earlier and later dates
    dates.sort(key=lambda x: x[1])

    # Display the extracted results
    st.subheader("Extracted Information:")

    if dates:
        st.write(f"**Date of Issue** : {dates[0][1].strftime('%Y/%m/%d')}")
        if len(dates) > 1:
            st.write(f"**Date of Expiry** : {dates[1][1].strftime('%Y/%m/%d')}")

    if passport_number:
        st.write(f"**Passport Number** : {passport_number}")

    if full_name:
        st.write(f"**Full Name** : {full_name}")
