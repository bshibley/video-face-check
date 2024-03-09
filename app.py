import cv2
import face_recognition
import numpy as np
import os
import chromadb
import pprint

debug = True
id_threshold = 0.3

# If chroma db doesn't exist, create it
if not os.path.exists('chroma'):
    os.makedirs('chroma')

# Load chroma database of known face embeddings
chroma_dir = 'chroma'
chroma_client = chromadb.PersistentClient(path=chroma_dir)
chroma_collection = chroma_client.get_or_create_collection(name="known_faces")

# Print names in the database
res = chroma_collection.get()
print('Names in the database:')
for name in res["ids"]:
    print(name)

# Ask user to enter their name
name = input('Enter your name: ')

# If user's name is not found in the chroma database, ask user
# to upload a photo of their identification and store their name
# and face embedding in the chroma database
res = chroma_collection.get(
    ids=[name],
)
if debug:
    print("Number of results: ", len(res["ids"]))
    pprint.pprint(res)
if len(res["ids"]) < 1:
    print('Name not found in chroma database')
    print('Please upload a photo of your identification')
    
    # Load the photo of the user's identification
    id_image_path = input('Enter the path to the photo of your identification: ')
    id_image = face_recognition.load_image_file(id_image_path)

    # Get the face encoding of the person in the photo and 
    # ensure that there is one face in the photo
    id_face_encodings = face_recognition.face_encodings(id_image)
    if debug:
        print('Number of faces found in the photo: ', len(id_face_encodings))
        pprint.pprint(id_face_encodings)
    if len(id_face_encodings) == 1:
        id_face_encoding = id_face_encodings[0].tolist()
        names = [name]
        embeddings = [id_face_encoding]
        if debug:
            print('Adding:')
            pprint.pprint(names)
            pprint.pprint(embeddings)
        chroma_collection.add(ids=names, embeddings=embeddings)
    elif len(id_face_encodings) == 0:
        print('No face found in the photo')
        exit()
    else:
        print('More than one face found in the photo')
        exit()

# Collect frames from webcam or video file and
# perform face recogntion to determine if the identified person is
# is present throughout the video
    
# Load the video and convert to np stack
vid_image_path = input('Enter the path to the video file: ')
cap = cv2.VideoCapture(vid_image_path)
frames = []
ret = True
while ret:
    ret, img = cap.read()
    if ret:
        frames.append(img)
cap.release()
cv2.destroyAllWindows()
video = np.stack(frames, axis=0)

# Get the face encoding of the person in the chroma database
baseline = chroma_collection.get(ids=[name],include=['embeddings'],)["embeddings"]

frame_count = 0
pos_id_frame_count = 0
neg_id_frame_count = 0

# For each 10th frame in the video
for frame in video:
    frame_count += 1
    if frame_count % 10 != 5:
        continue
    face_encodings = face_recognition.face_encodings(frame)
    for face_encoding in face_encodings:
        # Calculate the l2 distance between the face encoding of the person in the frame
        # and the saved face encoding
        distance = np.linalg.norm(np.array(baseline) - face_encoding, axis=1)/2
        #
        if distance < id_threshold:
            pos_id_frame_count += 1
            print(name + ' is present. Distance: ' + str(distance))
        else:
            neg_id_frame_count += 1
            res = chroma_collection.query(face_encoding.tolist(), n_results=1, 
                include=['distances','embeddings'],)
            print('Person not found. Distance: ' + str(distance) + '  Closest match: ' + res["ids"][0][0] + ' with distance: ' + str(res["distances"][0][0]))

# At the end of the video, display the total percentage of time the identified person
# was present in the video
print('Total time ' + name + ' was present: ' + str(pos_id_frame_count / (pos_id_frame_count+neg_id_frame_count) * 100) + '%')