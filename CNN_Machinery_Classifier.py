import os
import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D
# from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Flatten
from keras.layers import Dense


def getImagesPaths(TEST_N_TRAIN_IMG_PATH):
    dirs = np.array(os.listdir(TEST_N_TRAIN_IMG_PATH))
    # print(dirs)
    cwd = os.getcwd()
    # print(cwd)
    test_n_train_imgs_dirs = np.array([os.path.join(cwd, TEST_N_TRAIN_IMG_PATH, i).replace('\\', '/') for i in dirs])
    # print(test_n_train_imgs_dirs)
    files = np.array([os.listdir(i) for i in test_n_train_imgs_dirs])
    # print(files)
    test_n_train_imgs_paths = np.array(list(map(lambda dir, file: [
        [os.path.join(test_n_train_imgs_dirs[i], files[i][j]).replace('\\', '/') for j in range(len(files[0]))] for i in
        range(len(dirs))], dirs, files)))
    return test_n_train_imgs_paths[0]

def readImageBits(TEST_N_TRAIN_IMG_PATH):
    IMGS_PATHS = getImagesPaths(TEST_N_TRAIN_IMG_PATH)
    # imgs = cv.imread('./RSS/Image_Datasets/Air Cooler/cooler1.jpg')
    # imgs = cv.imread(IMGS_PATHS)



    # read_imgs = np.vectorize(lambda img: np.append(cv.imread(img), os.path.basename(img)))
    # imgs = read_imgs(IMGS_PATHS)
    # print(IMGS_PATHS)

    # print(IMGS_PATHS.shape)


    # print(IMGS_PATHS)


    # for i in IMGS_PATHS:
    #     print(i)

    machinery_class_names = [list(set([os.path.basename(os.path.dirname(i)) for i in j]))[0] for j in IMGS_PATHS]
    # print(machinery_class_names)


    # print(cv.imread(IMGS_PATHS[0, 0]))



    data = []

    for i in range(len(IMGS_PATHS)):
        for j in range(len(IMGS_PATHS[i])):
            # print(IMGS_PATHS[i, j])
            data.append(np.array([cv.imread(IMGS_PATHS[i, j]), machinery_class_names[i]]))
            # break
            # print(np.append(cv.imread(IMGS_PATHS[i, j]), machinery_class_names[i], axis=))

            # data = data + np.append(cv.imread(IMGS_PATHS[i, j]), machinery_class_names[i])

    np_data = np.array(data)
    # print(np_data)
    # print(np_data.shape)

    return np_data




    # imgs = list(map(lambda img: np.append(cv.imread(img), os.path.basename(img)), IMGS_PATHS))
    # imgs = list(map(lambda img: cv.imread(img), IMGS_PATHS))
    # print(imgs)


    # gr_img = cv.cvtColor(imgs, cv.COLOR_BGR2GRAY)
    # print(gr_img)


    # np_img = np.array(gr_img)


def machineryClassifier(IMG_PATH, TEST_N_TRAIN_IMG_PATH):
    data = readImageBits(TEST_N_TRAIN_IMG_PATH)

    # print(data.shape)
    # print(data)

    # X = data[:10, 0]
    # Y = data[10:20, 0]
    # Z = data[20:, 0]

    X = data[:, 0]
    # X = data[0, 0]
    Y = data[:, 1]
    # Y = data[0, 1]

    '''
    print('X =', X)
    print(X.shape)
    print('*' * 30)
    print('Y =', Y)
    print('*' * 30)
    '''
    # print('Z =', Z)

    # print(np.array(X).flatten())

    # plt.imshow(X[0], cmap='hot', interpolation='nearest')
    # plt.show()


    l_encode = LabelEncoder()

    # print(X.shape)
    # print(Y.shape)
    # print(X[0], Y[0, 0])
    # print(Y.flatten().flatten())

    x = X
    y = l_encode.fit_transform(Y)
    # print(y)
    # print(x)


    # Split the dataset into training and testing sets-
    rs = 42
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = rs)



    # Y =
    # print


    # print(np_img)



    # Encoding Dataset (Feature Scaling)-
    ss = StandardScaler()

    print(f'x_train = {x_train}\n***************************************\nx_test = {x_test}')

    print('*' * 30)
    x_train_flattened = np.array([i.flatten() for i in x_train], dtype = object)
    x_test_flattened = np.array([i.flatten() for i in x_test], dtype = object)

    print(f'x_train_flattened = {x_train_flattened}\n***************************************\nx_test_flattened = {x_test_flattened}')

    print(f'Shape of x_train_flattened = {x_train_flattened.shape} and Shape of x_test = {x_test_flattened.shape}')


    # x_train_flattened = np.array([i for i in x_train_flattened], dtype = object)
    # x_test_flattened = np.array([i for i in x_test_flattened], dtype = object)


    print(x_train_flattened, x_test_flattened)

    print(x_train_flattened[0], '\n', x_test_flattened[0])


    feats = x_train_flattened[0].size
    print('Feature count =', feats)

    x_train_scaled = np.array([ss.fit_transform(i.reshape(-1, 1)) for i in x_train_flattened], dtype = object)
    x_test_scaled = np.array([ss.transform(i.reshape(-1, 1)) for i in x_test_flattened], dtype = object)

    # x_train_scaled = np.array([ss.fit_transform(i) for i in x_train_flattened], dtype = object)
    # x_test_scaled = np.array([ss.transform(i) for i in x_test_flattened], dtype = object)

    # x_train_scaled = np.array(ss.fit_transform(x_train_flattened), dtype = object)
    # x_test_scaled = np.array(ss.transform(x_test_flattened), dtype = object)


    print(f'x_train_scaled = {x_train_scaled}\nx_test_scaled = {x_test_scaled}\ny_train = {y_train}\ny_test = {y_test}')


    #Training the Sequential Classifier-
    feats = 32
    CNN_seq_clf = Sequential([
        Conv2D(filters = feats, kernel_size = (3, 3), activation = 'relu', input_shape = (feats, feats, 3)),
        AveragePooling2D(pool_size = (2, 2)),
        Flatten(),
        Dense(units = 100, activation = 'relu'),
        Dense(units = 50, activation = 'relu'),
        Dense(units = 25, activation = 'softmax')
    ])


    CNN_seq_clf.compile(optimizer = 'adam', loss = SparseCategoricalCrossentropy, metrics = ['accuracy'])

    print('Sequential CNN Compilation Successful!')


    print('Shape of x_trained_scaled =', x_train_scaled.flatten().shape)
    print('Shape of x_test_scaled =', x_test_scaled.flatten().shape)
    print('Shape of y_train =', y_train.shape)
    print('Shape of y_test =', y_test.shape)

    print('Shape of x_trained_scaled[0] =', x_train_scaled[0].flatten().shape)
    print('Shape of x_test_scaled[0] =', x_test_scaled[0].flatten().shape)
    print('Shape of y_train[0] =', y_train[0].shape)
    print('Shape of y_test[0] =', y_test[0].shape)


    # x_train_scaled = np.asarray(x_train_scaled).astype(np.float32)

    # x_train_scaled = np.array(x_train_scaled[0].flatten(), dtype = float)

    print(x_train_scaled[0], '\n', y_train)

    model = CNN_seq_clf.fit(np.asarray(x_train_scaled, dtype = float), np.asarray(y_train, dtype = float), epochs = 500, batch_size = 64)
    # model = CNN_seq_clf.fit(x_train_scaled, y_train, epochs = 100, batch_size = 64)

    y_pred = CNN_seq_clf.predict(x_test_scaled)
    print('y_pred =', y_pred)



    # cv.imshow('GREY SCALED IMAGE', np_img)
    # cv.waitKey(0)


# readImageBits("RSS/Image_Datasets")
machineryClassifier("./RSS/Unknown_Machinery.jpg", "./RSS/Image_Datasets")


