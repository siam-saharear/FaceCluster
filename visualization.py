import cv2
import random
from image_utils import image_resizer



def draw_connections(canvas, matches, relocation_coordinates):
    for match in matches:
        image_path1, image_path2, bbox1, bbox2 = match

        image_1_bbox = relocation_coordinates[image_path1]
        i1x1, i1y1, i1x2, i1y2 = image_1_bbox
        b1x1, b1y1, b1x2, b1y2 = bbox1
        b1x1 = int(i1x1+b1x1)
        b1y1 = int(i1y1+b1y1)
        b1x2 = int(i1x1+b1x2)
        b1y2 = int(i1y1+b1y2)

        image_2_bbox = relocation_coordinates[image_path2]
        i2x1, i2y1, i2x2, i2y2 = image_2_bbox
        b2x1, b2y1, b2x2, b2y2 = bbox2
        b2x1 = int(i2x1+b2x1)
        b2y1 = int(i2y1+b2y1)
        b2x2 = int(i2x1+b2x2)
        b2y2 = int(i2y1+b2y2)

        color = tuple(random.randint(0,255) for _ in range(3) )
        cv2.rectangle(canvas, (b1x1,b1y1), (b1x2,b1y2), color, 20)
        cv2.rectangle(canvas, (b2x1,b2y1), (b2x2,b2y2), color, 20)
        cv2.line(canvas, (b1x1,b1y1), (b2x1,b2y1), color, 20)
    return canvas


def review_faces(face_database, drawn_canvas, relocation_coordinates):
    for person in face_database:
        person_id = person["id"]
        embedding = person["embedding"]
        appearances = person["appearance"]
        if len(appearances) <= 2:
            continue
        canvas_copy = drawn_canvas.copy()
        for appearance in appearances:
            path = appearance["path"]
            bbox = appearance["bbox"]
            x1,y1, x2,y2 = bbox
            face = appearance["face"]

            frame_coordinates = relocation_coordinates[path]
            a1,b1, a2,b2 = frame_coordinates
            cv2.rectangle(canvas_copy, (int(a1+x1), int(b1+y1)), (int(a1+x2), int(b1+y2)), (0,200,0), 50)
        canvas_copy = image_resizer(canvas_copy, 700)
        cv2.imshow(str(person_id), canvas_copy)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
        elif key == ord("n"):
            cv2.destroyAllWindows()
            continue
        else:
            break