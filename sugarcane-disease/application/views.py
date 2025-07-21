from django.shortcuts import render
from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login ,logout,authenticate
from django.shortcuts import get_object_or_404

# Create your views here.
def home(request):
    return render(request,'index.html')
def register(request):
    if request.method == 'POST':
        First_Name = request.POST['name']
        Email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        confirmation_password = request.POST['cnfm_password']
        if password == confirmation_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists, please choose a different one.')
                return redirect('register')
            else:
                if User.objects.filter(email=Email).exists():
                    messages.error(request, 'Email already exists, please choose a different one.')
                    return redirect('register')
                else:
                    user = User.objects.create_user(
                        username=username,
                        password=password,
                        email=Email,
                        first_name=First_Name,
                    )
                    user.save()
                    return redirect('login')
        else:
            messages.error(request, 'Passwords do not match.')
        return render(request, 'register.html')
    return render(request, 'register.html')
def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        if User.objects.filter(username=username).exists():
            user=User.objects.get(username=username)
            if user.check_password(password):
                user = authenticate(username=username,password=password)
                if user is not None:
                    login(request,user)
                    messages.success(request,'login successfull')
                    return redirect('/')
                else:
                   messages.error(request,'please check the Password Properly')
                   return redirect('login')
            else:
                messages.error(request,"please check the Password Properly")  
                return redirect('login') 
        else:
            messages.error(request,"username doesn't exist")
            return redirect('login')
    return render(request,'login.html')
# Load and preprocess the dataset
def logout_view(request):
    logout(request)
    return redirect('login')

#import require python classes and packages
import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
import pickle
from sklearn.metrics import confusion_matrix #class to calculate accuracy and other metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import keras
from keras import Model, layers
import pandas as pd
from keras.optimizers import SGD #import SGD class
from keras.optimizers import Adam #import Adam class optimizer
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
accuracy = []
precision = []
recall = []
fscore = []
alg=[]
X = []
Y = []
path = "Datasets"
labels = []

global X_train, X_test, y_train, y_test, X_val, y_val
extension_model=None
def loadDataset():
    global X, Y
    global path, labels, name, X, Y, root, img, getLabel
    #define function to load class labels
    for root, dirs, directory in os.walk(path):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name.strip())
    def getLabel(name):
        index = -1
        for i in range(len(labels)):
            if labels[i] == name:
                index = i
                break
        return index
    #loop and read all images from dataset
    if os.path.exists('model/X.txt.npy'):#if images already processed then load all images
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:#if not processed then read and process each image
        X = []
        Y = []
        for root, dirs, directory in os.walk(path):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])#read image
                    img = cv2.resize(img, (32, 32))#resize image
                    X.append(img)#addin images features to training array
                    label = getLabel(name)
                    Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y) 
def Augmentation():
    loadDataset()
    global X, Y
    # preprocess images like shuffling and normalization
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y_before = Y.copy()  # Save pre-augmentation Y for comparison
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    if os.path.exists('model/aug_X.txt.npy'):
        X = np.load('model/aug_X.txt.npy')
        Y = np.load('model/aug_Y.txt.npy')
    else:
        aug = ImageDataGenerator(rotation_range=15, shear_range=0.8, horizontal_flip=True)
        data = aug.flow(X_train, y_train, 1)
        X = []
        Y = []  
        for x, y in data:
            x = x[0]
            y = y[0]
            X.append(x)
            Y.append(y)
            if len(Y) > 30000:
                break    
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/aug_X.txt',X)
        np.save('model/aug_Y.txt',Y)
def DisplayComparison(Y_before):
    global labels
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define a bright, appealing color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5', '#9B59B6']
    # Ensure we cycle through colors if there are more bars than colors
    colors = colors * (len(labels) // len(colors) + 1)
    
    # Before Augmentation
    unique_before, count_before = np.unique(Y_before, return_counts=True)
    count_before_full = np.zeros(len(labels))
    for i, val in enumerate(unique_before):
        count_before_full[val] = count_before[i]
    
    y_pos = np.arange(len(labels))
    bars1 = ax1.bar(y_pos, count_before_full, color=colors[:len(labels)], edgecolor='black', linewidth=1.2)
    ax1.set_xticks(y_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax1.set_xlabel("Class Labels", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax1.set_title("Before Augmentation", fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # After Augmentation
    unique_after, count_after = np.unique(np.argmax(Y, axis=1), return_counts=True)
    count_after_full = np.zeros(len(labels))
    for i, val in enumerate(unique_after):
        count_after_full[val] = count_after[i]
    
    bars2 = ax2.bar(y_pos, count_after_full, color=colors[:len(labels)], edgecolor='black', linewidth=1.2)
    ax2.set_xticks(y_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax2.set_xlabel("Class Labels", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax2.set_title("After Augmentation", fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Customize the appearance
    fig.patch.set_facecolor('#F0F0F0')  # Light gray background for the figure
    ax1.set_facecolor('#FAFAFA')        # Slightly lighter background for subplot
    ax2.set_facecolor('#FAFAFA')
def splitDataset():
    Augmentation()
    global X, Y, X_train, X_test, y_train, y_test, X_val, y_val
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    
lodeddata=None            
def load_data(request):
    global X, Y, X_train, X_test, y_train, y_test, X_val, y_val
    global lodeddata
    splitDataset()
    output = (
            "Data loaded successfully for training the model"
            + "<br><strong>Train Data:</strong><br>"
            + "<strong>X_train Shape:</strong> " + str(X_train.shape)
            + "<br><strong>Y_train Shape:</strong> " + str(y_train.shape)
        )
    lodeddata=True
    return render(request, 'prediction.html', {'output': output})
def calculateMetrics(algorithm, predict, testY):  
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100     
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    alg.append(algorithm)
labels = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
def load_model(request):
    if lodeddata is None:
        data='please load the Dataset First'
        return render(request, 'prediction.html', {'output': data}) 
    global X, Y, X_train, X_test, y_train, y_test, X_val, y_val, eyenet_model,extension_model
    eyenet_model = keras.models.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
        keras.layers.Conv2D(filters=16, kernel_size=(9,9), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
        keras.layers.Conv2D(filters=8, kernel_size=(7,7), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
        keras.layers.Conv2D(filters=8, kernel_size=(6,6), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
        keras.layers.Conv2D(filters=8, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
        keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(2,2), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(y_train.shape[1], activation='softmax')])    
    #training CNN with Adam Optimizer
    opt = Adam(lr=0.001)#defining Adam
    eyenet_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])#compiling the model
    #compiling, training and loading the model
    if os.path.exists("model/adam_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/adam_weights.hdf5', verbose = 1, save_best_only = True)
        hist = eyenet_model.fit(X_train, y_train, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/adam_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        eyenet_model.load_weights("model/adam_weights.hdf5")
    #perfrom prediction on test data
    predict = eyenet_model.predict(X_val)
    predict = np.argmax(predict, axis=1)
    y_val1 = np.argmax(y_val, axis=1)
    acc = accuracy_score(y_val1, predict) * 100
    print("CNN Adam Validation Accuracy  :  "+str(acc))
    predict = eyenet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("Custom CNN Testing", predict, y_test1)    
    #proposed model Training
    extension_model = Sequential()
    extension_model.add(InputLayer(input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])))
    extension_model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
    extension_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    extension_model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    extension_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    extension_model.add(BatchNormalization())
    extension_model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    extension_model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    extension_model.add(BatchNormalization())
    extension_model.add(Flatten())
    extension_model.add(Dense(units=100, activation='relu'))
    extension_model.add(Dense(units=100, activation='relu'))
    extension_model.add(Dropout(0.25))
    extension_model.add(Dense(units=y_train.shape[1], activation='softmax'))
    extension_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    if os.path.exists("model/extension_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
        hist = extension_model.fit(X_train, y_train, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/extension_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        extension_model.load_weights("model/extension_weights.hdf5")
    #perfrom prediction on test data
    predict = extension_model.predict(X_val)
    predict = np.argmax(predict, axis=1)
    y_val1 = np.argmax(y_val, axis=1)
    acc = accuracy_score(y_val1, predict) * 100
    print("Extension with Adam & Valid Padding Validation Accuracy  :  "+str(acc))
    predict = extension_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("Proposed CNN Testing", predict, y_test1)   
    data={
        'algorithms':alg,
        'accuracy':accuracy,
        'precision':precision,
        'f1-score':fscore,
        'recall':recall
    }
    output = pd.DataFrame(data)
    return render(request, 'prediction.html', {'output': output.to_html(classes="table table-bordered")})
from django.core.files.storage import default_storage
def prediction(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)

        # Read and process the image
        image = cv2.imread(file_path)
        img = cv2.resize(image, (32, 32))
        im2arr = np.array(img).reshape(1, 32, 32, 3).astype('float32') / 255
        
        # Delete the file after reading
        default_storage.delete(file_path)

        # Predict using the model
        predict = eyenet_model.predict(im2arr)
        predict = np.argmax(predict)

        # Resize for display
        img = cv2.resize(image, (400, 300))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Add text
        cv2.putText(img, f'Predicted As: {labels[predict]}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Save the output image using Matplotlib
        output_path = 'static/images/output.png'
        plt.figure(figsize=(3,3))
        plt.imshow(img)
        plt.axis('off')  # Hide axis for better visualization
        plt.savefig(output_path)  # Only pass the file path

        return render(request, 'prediction.html', {'data': output_path, 'predict': labels[predict]})

    return render(request, 'prediction.html', {'test': True})