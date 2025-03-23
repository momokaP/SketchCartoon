import cv2
import numpy as np

def add_cartoon_edges(image_path, output_path):
    # 이미지 로드
    img = cv2.imread(image_path)

    # 1. 그레이스케일 변환 및 블러 처리 (노이즈 감소)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 1)  # 더 부드러운 엣지를 위해 medianBlur 사용

    # 2. Canny 엣지 검출 (더 정교한 설정)
    edges = cv2.Canny(gray, 30, 150)  # 임계값을 조절하여 불필요한 선 제거

    # 3. 적응형 임계값을 사용한 엣지 보정
    edges_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                           cv2.THRESH_BINARY, 9, 9)

    # 4. 두 개의 엣지를 결합
    refined_edges = cv2.bitwise_and(cv2.bitwise_not(edges), edges_adaptive)

    # 7. 엣지를 컬러 이미지로 변환 후 원본과 합성
    edges_colored = cv2.cvtColor(refined_edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(img, edges_colored)

    cartoon = cv2.pyrMeanShiftFiltering(cartoon, 10, 25)
    
    # 결과 저장 및 출력
    cv2.imwrite(output_path, cartoon)
    cv2.imshow("Cartoonized Image with Refined Edges", 
               cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 사용 예제
add_cartoon_edges("A.jpg", "output_A.jpg")