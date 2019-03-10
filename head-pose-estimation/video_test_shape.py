import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from pynput.mouse import Button, Controller
from scipy.spatial import distance as dist

mouse = Controller()

face_landmark_path = '../shape_predictor_68_face_landmarks.dat'

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


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


def main():
    # return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=512)
        if ret:
            face_rects = detector(frame, 0)

            if len(face_rects) > 0:
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle = get_head_pose(shape)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                for start, end in line_pairs:
                    cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

                cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)

                mouth = shape[face_utils.FACIAL_LANDMARKS_IDXS["mouth"][0]:face_utils.FACIAL_LANDMARKS_IDXS["mouth"][1]]
                # mar = smile(mouth)
                isMouthOpen = is_mouth_open(mouth)
                print('is mouth open? ', isMouthOpen)
                mouthHull = cv2.convexHull(mouth)
                # print(shape)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

                left_eye = shape[
                           face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][
                               1]]
                # left = smile(mouth)
                # print(left_eye)
                leftHull = cv2.convexHull(left_eye)
                # print(eye_aspect_ratio(left_eye))

                # print(shape)
                cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)

                right_eye = shape[
                            face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:
                            face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]
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

                # print(distance_between_eyes / (distance_eye_to_nose / distance_nose_to_mouth))

                if euler_angle[0, 0] < -5:
                    # print("UP")
                    y -= 5
                    #
                elif euler_angle[0,0] > 0:
                    # print("DOWN")
                    y += 5
                    # if euler_angle[0,0] > 10:
                    #     y += 5


                if euler_angle[1, 0] < -10:
                    # print("LEFT")
                    x -= 5
                    if euler_angle[1,0] < -20:
                        x -= 5
                elif euler_angle[1,0] > 10:
                    # print("RIGHT")
                    x += 5
                    if euler_angle[1,0] > 20:
                        x += 5

                # if items_within_percentage(distance_left_eye_to_nose, distance_right_eye_to_nose, 0.2):
                #     pass
                # elif items_within_percentage(distance_left_eye_to_nose, distance_right_eye_to_nose, 0.4):
                #     if distance_right_eye_to_nose > distance_left_eye_to_nose:
                #         print("LEFT")
                #         x -= 5
                #     else:
                #         print("RIGHT")
                #         x += 5
                # else:
                #     if distance_right_eye_to_nose > distance_left_eye_to_nose:
                #         print("LEFT")
                #         x -= 10
                #     else:
                #         print("RIGHT")
                #         x += 10

                if x != 0 or y != 0:
                    # print("Moving", x, y)
                    mouse.move(x, y)

            cv2.imshow("demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def items_within_percentage(num1, num2, percent):
    if num1 < num2:
        return num1 * (1 + percent) > num2
    else:
        return num2 * (1 + percent) > num1


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


def average_of_array(a):
    res = [0, 0]
    for item in a:
        res[0] += item[0]
        res[1] += item[1]
    return res[0] / len(a), res[1] / len(a)

def is_mouth_open(the_mouth):
    is_open = False

    A = dist.euclidean(the_mouth[3], the_mouth[9])
    B = dist.euclidean(the_mouth[2], the_mouth[10])
    C = dist.euclidean(the_mouth[4], the_mouth[8])
    avg = (A + B + C) / 3
    D = dist.euclidean(the_mouth[0], the_mouth[6])
    mar = avg / D

    print('mar', mar)
    threshold = 0.55
    if mar >= threshold:
        is_open = True
    return is_open


if __name__ == '__main__':
    main()
