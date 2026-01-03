from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import face_recognition
import os
import pickle
import random
import csv
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
UPLOAD_FOLDER = 'dataset'
ENCODINGS_FILE = 'trained_encodings.pkl'
ATTENDANCE_FILE = 'attendance.csv'
USER_DATA_FILE = 'user_data.csv'
EMAIL_ADDRESS = 'majorproject605@gmail.com'
EMAIL_PASSWORD = 'cqtn lydg cxrj jupc'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, 'rb') as f:
        known_encodings = pickle.load(f)
else:
    known_encodings = []

def send_confirmation_email(to_email, name, user_id, purpose="registration", date=None, time=None):
    if purpose == "registration":
        subject = "Registration Successful"
        body = f"Hello {name},\n\nYou have been successfully registered in the Face Recognition Attendance System.\nYour User ID is: {user_id}\n\nRegards,\nAdmin"
    elif purpose == "attendance":
        subject = "Attendance Marked"
        body = f"Hello {name} (User ID: {user_id}),\n\nYour attendance has been marked on {date} at {time}.\n\nRegards,\nAdmin"
    else:
        subject = "Notification"
        body = f"Hello {name} (User ID: {user_id}),\n\nThis is a system notification.\n\nRegards,\nAdmin"

    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"✅ Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")

def get_email_by_name(name):
    if not os.path.exists(USER_DATA_FILE):
        return None
    with open(USER_DATA_FILE, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == name:
                return row[1]
    return None

def get_user_id_by_name(name):
    if not os.path.exists(USER_DATA_FILE):
        return None
    with open(USER_DATA_FILE, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == name:
                return row[2]
    return None

@app.route('/')
def index():
    return render_template('index.html')

# REGISTRATION
@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    email = request.form['email']

    # Generate unique 5-digit user ID
    user_id = ''.join(random.choices('0123456789', k=5))

    user_folder = os.path.join(UPLOAD_FOLDER, name)
    os.makedirs(user_folder, exist_ok=True)

    image_file = request.files['image']
    base_path = os.path.join(user_folder, f"{name}_original.jpg")
    image_file.save(base_path)

    image = face_recognition.load_image_file(base_path)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        img_cv2 = cv2.imread(base_path)
        for i in range(40):
            copy_path = os.path.join(user_folder, f"{name}_{i}.jpg")
            cv2.imwrite(copy_path, img_cv2)

        known_encodings.append({'name': name, 'encoding': encodings[0]})
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(known_encodings, f)

        with open(USER_DATA_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, email, user_id])

        send_confirmation_email(email, name, user_id, purpose="registration")
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'no_face_detected'})

# MARK ATTENDANCE
@app.route('/mark', methods=['POST'])
def mark_attendance():
    section_open = request.form.get('section_open', 'false').lower() == 'true'
    if not section_open:
        return jsonify({'status': 'section_closed'})

    frame = request.files['frame'].read()
    np_img = np.frombuffer(frame, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    unknown_encodings = face_recognition.face_encodings(img)
    if not unknown_encodings:
        return jsonify({'status': 'no_face'})

    name_found = None
    best_distance = 1.0
    best_match = None

    for face_encoding in unknown_encodings:
        for data in known_encodings:
            distance = face_recognition.face_distance([data['encoding']], face_encoding)[0]
            if distance < 0.45 and distance < best_distance:
                best_distance = distance
                best_match = data['name']

    if best_match:
        name_found = best_match
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')

        # Check last attendance time
        recent_attendance = False
        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'r') as f:
                reader = csv.reader(f)
                for row in reversed(list(reader)):
                    if row[0] == name_found:
                        last_time = datetime.strptime(f"{row[1]} {row[2]}", '%Y-%m-%d %H:%M:%S')
                        if now - last_time < timedelta(minutes=30):
                            recent_attendance = True
                        break
        if recent_attendance:
            return jsonify({'status': 'recently_marked', 'name': name_found})

        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name_found, date_str, time_str])

        email = get_email_by_name(name_found)
        user_id = get_user_id_by_name(name_found)
        if email and user_id:
            send_confirmation_email(email, name_found, user_id, purpose="attendance", date=date_str, time=time_str)

        return jsonify({'status': 'marked', 'name': name_found})
    else:
        return jsonify({'status': 'unrecognized'})

# VIEW RECORD
@app.route('/records', methods=['GET'])
def records():
    data = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                name, date, time = row
                user_id = get_user_id_by_name(name)
                data.append({
                    'name': name,
                    'user_id': user_id if user_id else 'N/A',
                    'date': date,
                    'time': time
                })
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
