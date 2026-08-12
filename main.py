import os
import time
import random
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def scrape_image_paths(*args):
    folder_path = ""
    for arg in args:
        folder_path = os.path.join(folder_path, arg)
    files = os.listdir(folder_path)
    image_paths = []
    for file in files:
        file_extention = str((file.split("/")[-1]).split(".")[-1])
        if file_extention.lower() == "png" or file_extention.lower() == "jpg":
            image_paths.append(os.path.join(folder_path, file))
    return image_paths

def initailize_algo():
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1)
    return app

def detect_faces(image, app):
    faces = app.get(image)    
    return faces

def crate_canvas(image_paths, height, width):
    canvas = np.zeros((height,width,3), dtype=np.uint8)

    columns = int(np.ceil(np.sqrt(len(image_paths))))
    rows = int(np.ceil(len(image_paths) / columns))
    grid_size = (columns, rows)

    cell_height = canvas.shape[0]//rows
    cell_width = canvas.shape[1]//columns
    cell_size = (cell_height, cell_width)

    return canvas, grid_size, cell_size
 
def calculate_new_coordinates(bbox, conversion_scale):
    x1,y1,x2,y2 = bbox
    x1 = int(x1 *conversion_scale)
    y1 = int(y1 *conversion_scale)
    x2 = int(x2 *conversion_scale)
    y2 = int(y2 *conversion_scale)
    return np.array([x1,y1,x2,y2])

def resize_image(image, cell_height, cell_width, faces):
    h,w = image.shape[:2]
    conversion_scale = min(
        cell_width / w,
        cell_height / h
    )
    n_w = int(w * conversion_scale)
    n_h = int(h * conversion_scale)
    resized_image = cv2.resize(image, (n_w, n_h))

    recalculated_bbox = []
    for face in faces:
        bbox = face.bbox.astype(np.int32)
        n_bbox = calculate_new_coordinates(bbox, conversion_scale)
        face.bbox = n_bbox.astype(np.int32)
    return resized_image, faces

def image_distribution(canvas, images_n_faces, columns, cell_height, cell_width):  
    relocation_coordinates = {}
    for i,(image_path, data) in enumerate(images_n_faces.items()):
        image = data["image"]
        row = i//columns
        column = i%columns
        x = column*cell_width
        y = row*cell_height
        h,w = image.shape[:2]
        canvas[y:y+h, x:x+w] = image
        relocation_coordinates[image_path] = [x,y,x+w,y+h]
    return canvas, relocation_coordinates

def similarity(embedding_1, embedding_2, threshold=0.3):
    similrity = (np.dot(embedding_1, embedding_2)
                /
                (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
                )
    if similrity >= threshold:
        return True
    return False

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

def image_resizer(image, new_h=300):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    new_w = int(aspect_ratio*new_h)
    resized = cv2.resize(image, (new_w, new_h),)
    return resized

image_paths = scrape_image_paths("media", "mutual_face")

canvas, (columns,rows), (cell_height, cell_width) = crate_canvas(image_paths, 9000, 16000)

app = initailize_algo()

images_n_faces = {}
for image_path in image_paths:
    image = cv2.imread(image_path)
    all_faces = detect_faces(image, app)
    resized_image, all_faces_recalculated = resize_image(image, cell_height, cell_width, all_faces)
    images_n_faces[image_path] = {"image":resized_image, "faces":all_faces_recalculated}

drawn_canvas, relocation_coordinats = image_distribution(canvas, images_n_faces, columns, cell_height, cell_width)


face_database = []
face_id = 0
for image_path in image_paths:
    faces = images_n_faces[image_path]["faces"]
    for face in faces:
        face_embedding = face.embedding
        if len(face_database) == 0:
            face_database.append({"id":face_id,
                                  "embedding":face_embedding, 
                                  "appearance":[{"path":image_path, "bbox":face.bbox, "face":face}]})
            face_id += 1
            continue
        matched = False
        for person in face_database:
            compare_embedding = person["embedding"]
            if similarity(face_embedding, compare_embedding):
                person["appearance"].append({"path":image_path, "bbox":face.bbox, "face":face})
                matched = True
                break
        if not matched:
            face_database.append({"id":face_id,
                                    "embedding":face_embedding,
                                    "appearance":[{"path":image_path, "bbox":face.bbox, "face":face}]})
            face_id+=1

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

        frame_coordinates = relocation_coordinats[path]
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
                

