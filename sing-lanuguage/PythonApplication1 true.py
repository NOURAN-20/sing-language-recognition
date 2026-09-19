import os
dataset_path=r"C:\Users\CompuWay\OneDrive\Desktop\signlanguageproject\dataset"
dataset_info = {}

for item in os.listdir(dataset_path):
    full_path = os.path.join(dataset_path, item)
    if not os.path.isdir(full_path) or item.startswith('.'):
        continue
    image_count = len(os.listdir(full_path))
    dataset_info[item] = image_count

print(dataset_info)

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break
        
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
    cv2.imshow("Hand Landmarks", img)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

X = []
y = []

print("Starting feature extraction from dataset images...")

for label in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, label)
    if not os.path.isdir(class_path) or label.startswith('.'):
        continue
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                X.append(landmarks)
                y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Extraction finished! Total samples extracted: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import numpy as np
from sklearn.model_selection import train_test_split
np.save('X_data.npy',X)
np.save('y_data.npy',y)

print("data saved successfully")
X=np.load('X_data.npy')
y=np.load('y_data.npy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("data loded instantly")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Decision Tree Model
print("--- Training Decision Tree Model ---")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))

# 2. Random Forest Model
print("--- Training Random Forest Model ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

# 3. SVM Model (Support Vector Machine)
print("--- Training SVM Model ---")
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
svm_acc = accuracy_score(y_test, svm_model.predict(X_test))

# Final Comparison Summary
print("\n================== COMPREHENSIVE MODEL COMPARISON ==================")
print(f"1. Decision Tree Accuracy : {dt_acc * 100:.2f}%")
print(f"2. Random Forest Accuracy : {rf_acc * 100:.2f}%")
print(f"3. SVM Accuracy : {svm_acc * 100:.2f}%") 

best_acc = max(dt_acc, rf_acc, svm_acc)
if best_acc == rf_acc:
    print("\nConclusion: Random Forest achieved the highest accuracy and is selected as the final model.")
elif best_acc == svm_acc:
    print("\nConclusion: SVM achieved the highest accuracy and is selected as the final model.")
else:
    print("\nConclusion: Decision Tree achieved the highest accuracy and is selected as the final model.")
print("====================================================================")

import matplotlib.pyplot as plt
models = ['Decision Tree', 'Random Forest', 'SVM']
accuracies = [dt_acc * 100, rf_acc * 100, svm_acc * 100]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon'])

plt.title('Model Accuracy Comparison for Sign Language Recognition', fontsize=12)
plt.xlabel('Machine Learning Models', fontsize=10)
plt.ylabel('Accuracy (%)', fontsize=10)
plt.ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

plt.show()

print("Success! The comparison chart is displayed successfully.")
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("================== CLASSIFICATION REPORT ==================")
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print("===========================================================")

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest (Sign Language Recognition)', fontsize=12)
plt.xlabel('Predicted Label', fontsize=10)
plt.ylabel('True Label', fontsize=10)
plt.show()

print("Success! Evaluation metrics and Confusion Matrix generated successfully.")
plt.savefig('confusion_matrix.png')
print("Confusion Matrix saved as image successfully!")