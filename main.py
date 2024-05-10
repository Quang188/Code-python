!pip install --upgrade audioread
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os

data_path = "/content/drive/MyDrive/Train"

classes = ["guitar", "piano", "violin", "drum"]

data = []
labels = []

for label in classes:
    class_path = os.path.join(data_path, label)
    if os.path.isdir(class_path):
        filenames = os.listdir(class_path)
        if filenames:
            for filename in filenames:
                audio, sr = librosa.load(os.path.join(class_path, filename))
                target_length = 22050  
                audio = librosa.util.fix_length(audio, size=target_length)
                mfccs = librosa.feature.mfcc(y=audio, sr=sr)
                data.append(mfccs)
                labels.append(classes.index(label))
        else:
            print("Thư mục '{}' không có dữ liệu.".format(class_path))
    else:
        print("Thư mục '{}' không tồn tại.".format(class_path))
data = np.array(data)
labels = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=40)

model = SVC(max_iter=100)


X_train_avg = np.mean(X_train, axis=1)
X_test_avg = np.mean(X_test, axis=1)

model.fit(X_train_avg, y_train)

accuracy = model.score(X_test_avg, y_test)
print("Độ chính xác sau khi giảm chiều:", accuracy)


new_audio, sr = librosa.load("/content/drive/MyDrive/Test/262461__orangefreesounds__delayed-piano-loop-120-bpm.wav")

target_length = 22050
new_audio = librosa.util.fix_length(new_audio, size=target_length)


new_mfccs = librosa.feature.mfcc(y=new_audio, sr=sr)
new_mfccs = np.expand_dims(new_mfccs, axis=0)

new_mfccs_avg = np.mean(new_mfccs, axis=1)


prediction = model.predict(new_mfccs_avg)
predicted_class = classes[prediction[0]]

print("Nhạc cụ được dự đoán:", predicted_class)

