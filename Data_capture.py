import cv2
import os
import argparse

def Data_capture(args):
    ''' Function To record Data'''
    data_dir = args.data_dir
    mode = args.mode
    directory = os.path.join(data_dir,mode)
    cap = cv2.VideoCapture(0)
    if not os.path.exists(data_dir):
        os.makedirs(os.path.join(data_dir))
    if not os.path.exists(os.path.join(data_dir,mode)):
        os.makedirs(os.path.join(data_dir,mode))

    for i in range(3):
        if not os.path.exists(os.path.join(data_dir,mode , str(i))):
            os.makedirs(os.path.join(data_dir,mode , str(i)))

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        count = {
            'zero': len(os.listdir(os.path.join(directory ,"0"))),
            'one': len(os.listdir(os.path.join(directory ,"1"))),
            'two': len(os.listdir(os.path.join(directory ,"2"))),
        }

        cv2.putText(frame, "ZERO : " + str(count['zero']), (700, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
        cv2.putText(frame, "ONE : " + str(count['one']), (700, 90), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
        cv2.putText(frame, "TWO : " + str(count['two']), (700, 110), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
        cv2.rectangle(frame, (220 - 1, 9), (620 + 1, 419), (255, 0, 0), 1)

        roi = frame[10:410, 220:520]

        cv2.imshow("Frame", frame)

        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27:  # esc key
            break
        if interrupt & 0xFF == ord('0'):
            cv2.imwrite(os.path.join(directory , '0' ,str(count['zero']) + '.jpg'), roi)
        if interrupt & 0xFF == ord('1'):
            cv2.imwrite(os.path.join(directory , '1' ,str(count['one']) + '.jpg'), roi)
        if interrupt & 0xFF == ord('2'):
            cv2.imwrite(os.path.join(directory,'2',str(count['two'])+'.jpg'), roi)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Capture Utility')

    parser.add_argument("--data_dir", type=str, default='Motor_AI_Test',
                        help='path to Save  Dataset')
    parser.add_argument("--mode", type=str, default='Your_Set_Mode',
                        help='define Subfolder (train,test,valid)')
    args = parser.parse_args()
    Data_capture(args)