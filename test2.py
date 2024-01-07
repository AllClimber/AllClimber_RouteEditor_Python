import cv2
import numpy as np

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭된 픽셀의 색상을 가져옴
        colors = image[y, x, :]
        lower = np.array([max(0, c - 30) for c in colors], dtype="uint8")
        upper = np.array([min(255, c + 30) for c in colors], dtype="uint8")

        # 선택된 색상 범위 내의 픽셀에 대한 마스크 생성
        mask = cv2.inRange(image, lower, upper)

        # 흑백 이미지 생성
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        # 선택된 색상만 유지하고 나머지는 흑백으로 변환
        result = np.where(mask[:, :, None].astype(bool), image, gray_image)

        # 결과 표시
        cv2.imshow('Selected Color', result)

# 이미지 불러오기
image = cv2.imread('/Users/user/project/all_climber/route_edit_python/sampleImages/bouldering_wall_1.webp')
cv2.imshow('Original Image', image)

# 마우스 클릭 이벤트 연결
cv2.setMouseCallback('Original Image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
