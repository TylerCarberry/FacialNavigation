from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
import imutils
import numpy as np
import time
import dlib
import cv2
from pynput.mouse import Button, Controller

mouse = Controller()


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def average_of_array(a):
    res = [0, 0]
    for item in a:
        res[0] += item[0]
        res[1] += item[1]
    return res[0] / len(a), res[1] / len(a)


def items_within_percentage(num1, num2, percent):
    if num1 < num2:
        return num1 * (1 + percent) > num2
    else:
        return num2 * (1 + percent) > num1


def smile(the_mouth):
    if True:
        return 0
    A = dist.euclidean(the_mouth[3], the_mouth[9])
    B = dist.euclidean(the_mouth[2], the_mouth[10])
    C = dist.euclidean(the_mouth[4], the_mouth[8])
    avg = (A + B + C) / 3
    D = dist.euclidean(the_mouth[0], the_mouth[6])
    mar = avg / D
    return mar


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


COUNTER = 0
TOTAL = 0

shape_predictor = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)

fps = FPS().start()
cv2.namedWindow("test")

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=256)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[face_utils.FACIAL_LANDMARKS_IDXS["mouth"][0]:face_utils.FACIAL_LANDMARKS_IDXS["mouth"][1]]
        mar = smile(mouth)
        mouthHull = cv2.convexHull(mouth)
        # print(shape)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        left_eye = shape[
                   face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
        # left = smile(mouth)
        # print(left_eye)
        leftHull = cv2.convexHull(left_eye)
        # print(eye_aspect_ratio(left_eye))

        # print(shape)
        cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)

        right_eye = shape[
                    face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]
        # left = smile(mouth)
        rightHull = cv2.convexHull(right_eye)
        # print(shape)
        cv2.drawContours(frame, [shape], -1, (0, 255, 0), 1)

        if eye_aspect_ratio(left_eye) < 0.18 and eye_aspect_ratio(right_eye) < 0.18:
            mouse.press(Button.left)
            mouse.release(Button.left)

        nose = shape[
               face_utils.FACIAL_LANDMARKS_IDXS["nose"][0]:face_utils.FACIAL_LANDMARKS_IDXS["nose"][1]]
        # left = smile(mouth)
        noseHull = cv2.convexHull(nose)
        # print(shape)
        cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)

        # distance_left_eye_to_nose = average_of_array(left_eye)

        # print("LEFT:", average_of_array(left_eye))
        # print("NOSE:", average_of_array(nose))
        # print("RIGHT:",average_of_array(right_eye))
        # print("------")

        distance_left_eye_to_nose = average_of_array(left_eye)[0] - average_of_array(nose)[0]
        distance_right_eye_to_nose = average_of_array(nose)[0] - average_of_array(right_eye)[0]
        distance_between_eyes = average_of_array(left_eye)[0] - average_of_array(right_eye)[0]

        average_of_eyes = average_of_array(left_eye + right_eye)
        average_of_nose = nose[6]  # average_of_array(nose)
        average_of_mouth = average_of_array(mouth)

        distance_eye_to_nose = average_of_eyes[1] - average_of_nose[1]
        distance_nose_to_mouth = average_of_mouth[1] - average_of_nose[1]

        x, y = 0, 0

        print(distance_between_eyes / (distance_eye_to_nose / distance_nose_to_mouth))

        # if distance_eye_to_nose / distance_nose_to_mouth < 3:
        #     print("UP")
        #     y -= 5
        # else:
        #     print("DOWN")
        #     y += 5

        if items_within_percentage(distance_left_eye_to_nose, distance_right_eye_to_nose, 0.2):
            pass
        elif items_within_percentage(distance_left_eye_to_nose, distance_right_eye_to_nose, 0.4):
            if distance_right_eye_to_nose > distance_left_eye_to_nose:
                print("LEFT")
                x -= 5
            else:
                print("RIGHT")
                x += 5
        else:
            if distance_right_eye_to_nose > distance_left_eye_to_nose:
                print("LEFT")
                x -= 10
            else:
                print("RIGHT")
                x += 10

        if x != 0 or y != 0:
            print("Moving", x, y)
            mouse.move(x, y)

        if mar <= .3 or mar > .38:
            COUNTER += 1
        else:
            if COUNTER >= 15:
                TOTAL += 1
                frame = vs.read()
                time.sleep(.3)
                frame2 = frame.copy()
                img_name = "opencv_frame_{}.png".format(TOTAL)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
            COUNTER = 0

        cv2.putText(frame, "MAR: {}".format(mar), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    fps.update()

    key2 = cv2.waitKey(1) & 0xFF
    if key2 == ord('q'):
        break

fps.stop()

cv2.destroyAllWindows()
vs.stop()
