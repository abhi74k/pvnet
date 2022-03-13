import cv2
import matplotlib.pyplot as plt
import numpy as np

import pvnet_utils


def label_points_for_drawing(image_points):  # (b, f = back, front), (l, r = left, right), (u, d = up , down)
    labeled_points = {}
    labeled_points['bld'] = (int(round(image_points[0][0])), int(round(image_points[0][1])))
    labeled_points['blu'] = (int(round(image_points[1][0])), int(round(image_points[1][1])))
    labeled_points['fld'] = (int(round(image_points[2][0])), int(round(image_points[2][1])))
    labeled_points['flu'] = (int(round(image_points[3][0])), int(round(image_points[3][1])))
    labeled_points['brd'] = (int(round(image_points[4][0])), int(round(image_points[4][1])))
    labeled_points['bru'] = (int(round(image_points[5][0])), int(round(image_points[5][1])))
    labeled_points['frd'] = (int(round(image_points[6][0])), int(round(image_points[6][1])))
    labeled_points['fru'] = (int(round(image_points[7][0])), int(round(image_points[7][1])))
    return labeled_points


def draw_bounding_box(img, labeled_points, colour=(255, 0, 0)):
    cv2.line(img, labeled_points['bld'], labeled_points['blu'], colour, 2)
    cv2.line(img, labeled_points['bld'], labeled_points['fld'], colour, 2)
    cv2.line(img, labeled_points['bld'], labeled_points['brd'], colour, 2)
    cv2.line(img, labeled_points['blu'], labeled_points['flu'], colour, 2)
    cv2.line(img, labeled_points['blu'], labeled_points['bru'], colour, 2)
    cv2.line(img, labeled_points['fld'], labeled_points['flu'], colour, 2)
    cv2.line(img, labeled_points['fld'], labeled_points['frd'], colour, 2)
    cv2.line(img, labeled_points['flu'], labeled_points['fru'], colour, 2)
    cv2.line(img, labeled_points['fru'], labeled_points['bru'], colour, 2)
    cv2.line(img, labeled_points['fru'], labeled_points['frd'], colour, 2)
    cv2.line(img, labeled_points['frd'], labeled_points['brd'], colour, 2)
    cv2.line(img, labeled_points['brd'], labeled_points['bru'], colour, 2)


def visualize_pose(test_sample, image_points_pred):
    image_points_gt = test_sample['obj_keypoints_xy'][1:9, :]
    print(f'\nGround truth image points:\n {image_points_gt}')

    print(f'\nPredicted image points:\n {image_points_pred}')

    image = pvnet_utils.denormalize_img(test_sample['img']).permute(1, 2, 0).detach().numpy()
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    draw_bounding_box(img, label_points_for_drawing(image_points_gt))  # drawing ground truth bounding box in blue
    draw_bounding_box(img, label_points_for_drawing(image_points_pred),
                      (0, 255, 0))  # drawing pred bounding box in green

    plt.figure(figsize=(10, 10))
    plt.imshow(np.squeeze(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    plt.show()
