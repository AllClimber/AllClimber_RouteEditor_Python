import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class ImageAdjustor:
    # image 는 HSV 색체계여야 한다.
    def adjust_brightness(self, image, brightness):
        image[:, :, 2] = np.clip(image[:, :, 2] + brightness, 0, 255)
        return image
    
    # image 는 HSV 색체계여야 한다.
    def adjust_saturation(self, image, satruation):
        image[:, :, 1] = np.clip(image[:, :, 1] + satruation, 0, 255)
        return image

    # 평활화 (노이즈 감소, 색의 경계를 부드럽게 하면서 일관된 영역은 강조)
    def shift(self, image, sp, sr):
        return cv2.pyrMeanShiftFiltering(image, sp=sp, sr=sr)
    
    # 블러 처리 (노이즈를 제거)
    def blur(self, image, ksize, sigmaX):
        return cv2.GaussianBlur(image, ksize, sigmaX)

    # 엣지 강조 (3*3 단위의 커널을 이미지에 적용하여 미분하여 경계를 강조한다.)
    def enhance_edges(self, image):
        kernel = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]])
        edges = cv2.filter2D(image, -1, kernel)
        return cv2.addWeighted(image, 1, edges, 1.5, 0)

    def extract_holds(self, image):
        # 컨투어 추출 (윤곽선을 추출해서 연결된 객체로 인식되는 것들을 감싸는 사각형을 만드는 듯?)
        edges = cv2.Canny(image, 50, 150, L2gradient=False)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_size = 15
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_size]
        return [cv2.boundingRect(contour) for contour in filtered_contours]

    def indicate_holds(self, image, holds):
        # 원본이미지에 컨투어 표시
        hold_indicated_image = image.copy()
        for hold in holds:
            cv2.rectangle(hold_indicated_image, (hold[0], hold[1]), (hold[0] + hold[2], hold[1] + hold[3]), (0, 255, 0), 2)
        return hold_indicated_image

    def resize(self, image, target_width):
        # resize
        width = image.shape[1]
        height = image.shape[0]
        scale_factor = target_width / width
        target_height = int(height*scale_factor)
        return cv2.resize(image, (target_width, target_height), cv2.COLOR_BGR2RGB)

    def adjust_image(self, original_image, brightness, satruation):
        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        brightness_image = self.adjust_brightness(hsv_image, brightness)
        saturation_image = self.adjust_saturation(brightness_image, satruation)
        shifted_image = self.shift(saturation_image, 15, 50)

        # 그레이스케일로 변환 (픽셀당 하나의 값만을 가지므로 색상 정보를 유지하면서도 연산량을 감소시킴)
        bgrImage = cv2.cvtColor(shifted_image, cv2.COLOR_HSV2BGR)
        gray_image = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2GRAY)
        blurred_image = self.blur(gray_image, (1, 1), 0)
        edge_enhanced_image = self.enhance_edges(blurred_image)
        holds = self.extract_holds(edge_enhanced_image)
        hold_indicated_image = self.indicate_holds(original_image, holds)

        return np.concatenate((
            cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(brightness_image, cv2.COLOR_HSV2RGB),
            cv2.cvtColor(saturation_image, cv2.COLOR_HSV2RGB),
            cv2.cvtColor(shifted_image, cv2.COLOR_HSV2RGB),
            cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(edge_enhanced_image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(hold_indicated_image, cv2.COLOR_BGR2RGB)
        ), axis=1)

class ImageAdjustmentApp:

    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Image Adjustment App")

        self.original_image = cv2.imread(image_path)

        self.create_widgets()
        self.update_images()

    def create_widgets(self):
        self.processed_image_label = ttk.Label(self.root)
        self.processed_image_label.pack()

        brightness_label = ttk.Label(self.root, text="Brightness")
        brightness_label.pack()
        self.brightness_slider = ttk.Scale(self.root, from_=-100, to=100)
        self.brightness_slider.set(0)
        self.brightness_slider.pack()

        saturation_label = ttk.Label(self.root, text="Saturation")
        saturation_label.pack()
        self.saturation_slider = ttk.Scale(self.root, from_=-100, to=100)
        self.saturation_slider.set(0)
        self.saturation_slider.pack()

        # Update 버튼 추가
        update_button = ttk.Button(self.root, text="Update", command=self.update_images)
        update_button.pack()

    def adjust_image(self):
        image_adjuster = ImageAdjustor()
        brightness_value = self.brightness_slider.get()
        saturation_value = self.saturation_slider.get()
        adjusted_image_set = image_adjuster.adjust_image(self.original_image, brightness_value, saturation_value)
        return image_adjuster.resize(adjusted_image_set, 1600)

    def update_images(self):
        final_image = self.adjust_image()

        # Show final image
        final_image = Image.fromarray(final_image)
        photo = ImageTk.PhotoImage(image=final_image)
        self.processed_image_label.config(image=photo)
        self.processed_image_label.image = photo

# GUI application execution
root = tk.Tk()
image_path = "sampleImages/bouldering_wall_1.webp"
app = ImageAdjustmentApp(root, image_path)
root.mainloop()
