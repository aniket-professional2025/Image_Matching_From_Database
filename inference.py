# Importing the required function from the main.py
from main import find_closest_match
import matplotlib.pyplot as plt
import cv2

# Test on an image
query_image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Rehou_Image_Search\Query\QueryImg1.jpg"
closest_match_path, distance = find_closest_match(query_image_path)

# Printing the Results
if closest_match_path and distance is not None:
    print(f"Closest Match: {closest_match_path}")
    print(f"Distance: {distance:.4f}")
else:
    print("No valid match found.")

# Show the Image from Code Script
img = cv2.imread(closest_match_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.show()