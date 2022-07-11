import cv2 as cv
import pyautogui as pag


def identifyMachinery(IMG_PATH):
    try:
        img = cv.imread(IMG_PATH)
        # print(img.shape)
        h, w, ch = img.shape
        # img_resized = cv.resize(img, (0, 0), None, 0.5, 0.5)

        img_resized = cv.resize(img, (0, 0), None, 0.3, 0.3)

        # img_resized = cv.resize(img, (0, 0), None, 4, 4)

        win_name = 'MACHINERY'
        cv.namedWindow(win_name)
        W, H = pag.size()
        # print(w, h)
        # cv.moveWindow(win_name, w // 4, h // 4)
        cv.moveWindow(win_name, (2 * W + w) // 8, H // 4)



        gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
        # blur = cv.GaussianBlur(gray, (5, 5), 0)
        # thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]


        x, y, w, h = cv.boundingRect(gray)
        cv.rectangle(img_resized, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv.putText(img_resized, f'(Width={w}px, Height={h}px)', (x + w // 8, y + h // 2), cv.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
        #
        # cv.imshow("thresh", thresh)

        cv.imshow(win_name, img_resized)
        cv.waitKey(0)
        cv.destroyAllWindows()
    except(Exception,):
        print('Oops! Not a Valid Image/Path. Please try again...')
        IMG_PATH = input('Again Enter the Image Location: ')
        identifyMachinery(IMG_PATH)

# IMG_PATH = input('Please Enter the Image's Path/Location: ')
IMG_PATH = "./RSS/Unknown_Machinery.jpg"
identifyMachinery(IMG_PATH)


