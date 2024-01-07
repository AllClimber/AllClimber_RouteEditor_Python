import cv2
import numpy as np
from tkinter import *
from tkinter.ttk import *
from tkinter import ttk

class ImageAdjustmentApp:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Image Adjustment App")

        # 이미지 읽기
        self.original_image = cv2.imread(image_path)
        self.adjusted_image = self.original_image.copy()

        # 명도, 채도 초기값 설정
        self.brightness_value = 0
        self.saturation_value = 0

        # GUI 구성
        self.create_widgets()

    def create_widgets(self):
        # 이미지 라벨
        self.image_label = ttk.Label(self.root)
        self.image_label.pack()

        # 명도 조절 슬라이더
        brightness_label = ttk.Label(self.root, text="Brightness")
        brightness_label.pack()
        self.brightness_slider = ttk.Scale(self.root, from_=-100, to=100, command=self.adjust_image)
        self.brightness_slider.set(self.brightness_value)
        self.brightness_slider.pack()

        # 채도 조절 슬라이더
        saturation_label = ttk.Label(self.root, text="Saturation")
        saturation_label.pack()
        self.saturation_slider = ttk.Scale(self.root, from_=-100, to=100, command=self.adjust_image)
        self.saturation_slider.set(self.saturation_value)
        self.saturation_slider.pack()

        # 초기 이미지 업데이트
        self.update_image()

# 이미지 읽기
image_path = "/Users/user/project/all_climber/route_edit_python/sampleImages/bouldering_wall_1.webp"
image = cv2.imread(image_path)

shifted_image = cv2.pyrMeanShiftFiltering(image, sp=15, sr=50)

# 이미지 크기 조절 (원하는 크기로 조절할 수 있음)
# resized_image = cv2.resize(image, (new_width, new_height))
hsv_image = cv2.cvtColor(shifted_image, cv2.COLOR_BGR2HSV)

# 채도 증가를 위해 Saturation 채널에 상수를 더합니다.
saturation_increase = 10  # 조절할 채도 증가 값
hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] + saturation_increase, 0, 255).astype(np.uint8)

# 색조 증가를 위해 Hue 채널에 상수를 더합니다.
hue_increase = 20  # 조절할 색조 증가 값
hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_increase) % 180

brightness_increase = 50  # 조절할 명도 증가 값
brightened_image = np.clip(hsv_image + brightness_increase, 0, 255).astype(np.uint8)


# 그레이스케일 변환
gray_image = cv2.cvtColor(brightened_image, cv2.COLOR_BGR2GRAY)

# 콘트라스트 증가를 위해 히스토그램 평활화 적용
equalized_image = cv2.equalizeHist(gray_image)

# 이미지 블러 적용
blurred_image = cv2.GaussianBlur(gray_image, (1, 1), 0)

# 케니 엣지 검출
edges = cv2.Canny(blurred_image, 50, 150,L2gradient=False)

# 컨투어 찾기
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 작은 컨투어 제거
min_contour_size = 15  # 예시 크기, 조절 가능
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_size]

# 홀드의 경계를 찾아 리스트에 추가
holds = []
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    holds.append((x, y, x + w, y + h))

# 홀드 경계 그리기
final_image = image.copy()
for hold in holds:
    cv2.rectangle(final_image, (hold[0], hold[1]), (hold[2], hold[3]), (0, 255, 0), 2)

# 결과 이미지 출력
combined_image = np.concatenate((image, hsv_image, cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR), cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), final_image), axis=1)

cv2.imshow("Combined Images", combined_image)

if __name__ == "__main__":
    from PIL import Image, ImageTk

    # 이미지 경로
    image_path = "example_image.jpg"

    # GUI 애플리케이션 실행
    root = ttk.Tk()
    app = ImageAdjustmentApp(root, image_path)
    root.mainloop()

cv2.waitKey(0)
cv2.destroyAllWindows()
