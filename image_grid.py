import cv2
import numpy as np

def create_canvas(image_paths, height, width):
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
