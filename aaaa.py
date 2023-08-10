import cv2
path = "data/stable_diffusion_inferences/ukj/1440/1be3baf68e758ba5998e530b9f3d5842a5d1d647d694b24e7ddb3c1c46850a96/tuning/13933__gscale_10__nsteps_50_51_52_53_54.png"


img = cv2.imread(path)
h, w, c = img.shape

cv2.imwrite("data/stable_diffusion_inferences/ukj/1440/1be3baf68e758ba5998e530b9f3d5842a5d1d647d694b24e7ddb3c1c46850a96/tuning/13933__gscale_10__nsteps_53.png",
            img[:, 3*h:4*h, :])
