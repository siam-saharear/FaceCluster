from insightface.app import FaceAnalysis


def initialize_algo():
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1)
    return app

def detect_faces(image, app):
    faces = app.get(image)    
    return faces
