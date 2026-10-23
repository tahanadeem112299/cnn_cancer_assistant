import cv2 

image=cv2.imread("malignant.png")
print("DEtails: ",image.shape)
print("size: ",image.size)
print("dtype: ",image.dtype)
print("--------------------------------------\n")
resize=cv2.resize(image, (224,224))
print("size: ",resize.shape)
print("dtype: ",resize.dtype)
cv2.imshow("image",image)
cv2.imshow("resize",resize)
cv2.waitKey(0)
cv2.destroyAllWindows()