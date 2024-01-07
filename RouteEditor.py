import cv2
import numpy as np

def extract_holds(image_path, hold_color_lower, hold_color_upper):
    # 이미지를 읽어옴
    image = cv2.imread(image_path)

    # BGR을 HSV로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 지정한 범위 내의 색상만 유지
    mask = cv2.inRange(hsv, hold_color_lower, hold_color_upper)
    holds_colored = cv2.bitwise_and(image, image, mask=mask)

    # 그 외의 부분을 흑백으로 처리
    holds_gray = cv2.cvtColor(holds_colored, cv2.COLOR_BGR2GRAY)

    # 흑백 이미지에서 홀드의 경계 검출
    edges = cv2.Canny(holds_gray, 30, 100)

    # 경계 추출
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 홀드 정보를 담을 리스트
    holds_info = []

    for contour in contours:
        # 너무 작은 객체는 무시
        if cv2.contourArea(contour) > min_contour_area:
            # 경계 상자를 추출
            x, y, w, h = cv2.boundingRect(contour)

            # 홀드 색상 추출
            hold_roi = holds_colored[y:y+h, x:x+w]
            avg_color = np.mean(hold_roi, axis=(0, 1)).astype(int)
            representative_color = tuple(avg_color.tolist())

            # 홀드 정보를 리스트에 추가
            holds_info.append({'x': x, 'y': y, 'width': w, 'height': h, 'color': representative_color})

    return holds_info, holds_gray

# 이미지 파일 경로
image_path = '/Users/user/project/all_climber/route_edit_python/sampleImages/bouldering_wall_1.webp'

# 원하는 홀드의 색상 범위 지정 (HSV 값으로)
hold_color_lower = np.array([30, 10, 30], dtype=np.uint8)
hold_color_upper = np.array([50, 50, 50], dtype=np.uint8)

# 최소 경계 넓이
min_contour_area = 100

# 홀드 정보 추출 및 색상 추출
holds_info, holds_gray = extract_holds(image_path, hold_color_lower, hold_color_upper)

# 색상이 추출된 홀드 정보 출력
for i, hold in enumerate(holds_info):
    print(f"Hold {i + 1}: {hold['color']}")

# 특정 색상의 홀드만 남기고 나머지는 흑백처리
desired_color = (40, 20,20)  # 예시로 초록색 (BGR 순서)
filtered_holds_gray = np.zeros_like(holds_gray)

for hold in holds_info:
    if hold['color'] == desired_color:
        x, y, w, h = hold['x'], hold['y'], hold['width'], hold['height']
        filtered_holds_gray[y:y+h, x:x+w] = holds_gray[y:y+h, x:x+w]

# 결과 표시
cv2.imshow('Filtered Holds', filtered_holds_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
