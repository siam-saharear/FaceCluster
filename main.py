import cv2

from image_utils import scrape_image_paths
from face_detection import initialize_algo, detect_faces
from image_grid import crate_canvas, resize_image, image_distribution
from face_clustering import build_face_database
from visualization import review_faces




image_paths = scrape_image_paths("media", "mutual_face")

canvas, (columns,rows), (cell_height, cell_width) = crate_canvas(image_paths, 9000, 16000)

app = initialize_algo()

images_n_faces = {}
for image_path in image_paths:
    image = cv2.imread(image_path)
    all_faces = detect_faces(image, app)
    resized_image, all_faces_recalculated = resize_image(image, cell_height, cell_width, all_faces)
    images_n_faces[image_path] = {"image":resized_image, "faces":all_faces_recalculated}

drawn_canvas, relocation_coordinates = image_distribution(canvas, images_n_faces, columns, cell_height, cell_width)

face_database = build_face_database(image_paths, images_n_faces)

review_faces(face_database, drawn_canvas, relocation_coordinates)
