import cv2 as cv
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

# Set up the torchvision model
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.to(device)
model.eval()
preprocess = weights.transforms()

# Source the video feed
cap = cv.VideoCapture('resources/soccer_juggle.mp4')
count = 0

# While there is video, run the model per frame of the video
while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Resize & Convert the frame to a tensor
    frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    img = torch.from_numpy(cv.cvtColor(frame, cv.COLOR_BGR2RGB)).permute(2,0,1).to(device)

    # Predict the objects in the tensor
    batch = [preprocess(img)]
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4)
    
    # Convert tensor to png and display it every 10 passes
    if count == 10:
        im = to_pil_image(box.detach())
        im.show()
        count -= 10
    print(count)

    # Show the current frame through CV
    cv.imshow("Computer Vision", frame)
    key = cv.waitKey(1)

# Release the video and close the window
cap.release()
cv.destroyAllWindows()