from pharec import Pharec

image_size = (256, 512) # height, width
model_path = "models/2021-12-11_18;40;33.772059_wpd2_valacc0.9462_e8_b16.tf"
image_path = "collected_images/img_google.com_mzq9nrvk.png" 

pharec = Pharec(model_path, image_size)
image = pharec.load_image(image_path)
print("Image loaded... : image shape", image.shape)
pred_domain, pred_conf = pharec.predict_domain(image)
print("Predicted domain:", pred_domain, "with confidence:", pred_conf)

