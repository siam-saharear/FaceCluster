import numpy as np
import config

def similarity(embedding_1, embedding_2, threshold=config.FACE_SIMILARITY_THRESHOLD):
    similrity = (np.dot(embedding_1, embedding_2)
                /
                (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
                )
    if similrity >= threshold:
        return True
    return False


def build_face_database(image_paths, images_n_faces):
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
    return face_database