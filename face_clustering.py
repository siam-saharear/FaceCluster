import numpy as np
import config

def similarity(embedding_1, embedding_2, threshold=None):
    if threshold == None:
        threshold == config.FACE_SIMILARITY_THRESHOLD
    similrity = (np.dot(embedding_1, embedding_2)
                /
                (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
                )
    if similrity >= threshold:
        return True
    return False


def find_matching_person(face_embedding, face_database):
    for person in face_database:
        if similarity(face_embedding, person["embedding"]):
            return person
        else:
            return None 


def build_face_database(image_paths, images_n_faces):
    face_database = []
    face_id = 0
    for image_path in image_paths:
        faces = images_n_faces[image_path]["faces"]
        for face in faces:
            person = find_matching_person(face.embedding,face_database)
            appearance = {"path":image_path,
                          "bbox": face.bbox,
                          "face": face}
            if person:
                person["appearance"].append(appearance)
            else:
                face_database.append({
                    "id":face_id,
                    "embedding":face.embedding,
                    "appearance":[appearance]
                })
                face_id += 1
    return face_database