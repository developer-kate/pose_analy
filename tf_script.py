import cv2
import mediapipe as mp
import json
import os
import numpy as np
from collections import Counter
from datetime import timedelta

def extract_skeleton_data(video_path, output_json_path, visualize=False):
    """
    비디오 파일에서 모든 프레임의 관절 좌표를 추출하여 JSON 파일로 저장합니다.
    선택적으로 추출된 스켈레톤을 비디오 위에 그려 시각화합니다.

    Args:
        video_path (str): 분석할 비디오 파일의 경로.
        output_json_path (str): 관절 좌표를 저장할 JSON 파일의 경로.
        visualize (bool): 추출된 스켈레톤을 비디오 위에 그려서 보여줄지 여부.
    """

    # MediaPipe Pose 솔루션 초기화
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 비디오 파일을 열 수 없습니다. {video_path}")
        return

    frame_data_list = [] # 모든 프레임의 데이터를 저장할 리스트
    frame_count = 0

    print(f"'{video_path}' 비디오에서 스켈레톤 데이터 추출 중...")

    # MediaPipe Pose 모델 컨텍스트 매니저 사용
    with mp_pose.Pose(
        min_detection_confidence=0.5, # 포즈 감지 최소 신뢰도
        min_tracking_confidence=0.5) as pose: # 포즈 추적 최소 신뢰도

        while cap.isOpened():
            ret, frame = cap.read() # 프레임 읽기
            if not ret: # 더 이상 읽을 프레임이 없으면 종료
                break

            frame_count += 1
            # 성능 향상을 위해 이미지를 읽기 전 BGR을 RGB로 변환합니다.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False # 이미지 쓰기 불가능 설정 (처리 속도 향상)
            results = pose.process(image) # 포즈 처리

            # 이미지를 다시 BGR로 변환합니다. (OpenCV 표시를 위해)
            image.flags.writeable = True # 이미지 쓰기 가능 설정
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            frame_landmarks = [] # 현재 프레임의 랜드마크 데이터를 저장할 리스트
            if results.pose_landmarks: # 포즈 랜드마크가 감지된 경우
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    # 각 랜드마크의 ID, x, y, z 좌표 및 가시성(visibility) 저장
                    frame_landmarks.append({
                        "id": idx,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })
                
                if visualize: # 시각화 옵션이 True인 경우
                    # 이미지 위에 포즈 랜드마크와 연결선 그리기
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # 현재 프레임의 데이터를 전체 리스트에 추가
            frame_data_list.append({
                "frame_id": frame_count,
                "landmarks": frame_landmarks
            })

            if visualize:
                cv2.imshow('MediaPipe Pose', image) # 시각화 창에 이미지 표시
                if cv2.waitKey(5) & 0xFF == 27: # 5ms 대기 후 ESC 키(ASCII 27)를 누르면 종료
                    break

    cap.release() # 비디오 캡처 객체 해제
    if visualize:
        cv2.destroyAllWindows() # 모든 OpenCV 창 닫기

    # JSON 파일로 저장
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir) # 출력 디렉터리가 없으면 생성

    with open(output_json_path, 'w', encoding='utf-8') as f:
        # JSON 데이터를 파일에 쓰기 (한글 깨짐 방지, 들여쓰기 4칸)
        json.dump(frame_data_list, f, ensure_ascii=False, indent=4)

    print(f"스켈레톤 데이터가 '{output_json_path}'에 성공적으로 저장되었습니다.")
    
    # 텍스트 파일 생성
    generate_work_description(frame_data_list, output_json_path)

def generate_work_description(frame_data_list, json_path):
    """
    스켈레톤 데이터를 분석하여 작업 설명 텍스트 파일을 생성합니다.
    """
    if not frame_data_list:
        print("스켈레톤 데이터가 없어 텍스트 파일을 생성할 수 없습니다.")
        return
    
    # 설정
    FPS = 30
    SEGMENT_SEC = 7
    
    POSE_LANDMARKS = {
        'nose': 0, 'left_eye': 2, 'right_eye': 5, 'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24
    }
    
    def get_landmark(frame, idx):
        for lm in frame.get('landmarks', []):
            if lm['id'] == idx:
                return lm
        return None
    
    def analyze_segment(frames):
        arms, hands, torso, head, precision = [], [], [], [], 0
        for f in frames:
            l_sh = get_landmark(f, POSE_LANDMARKS['left_shoulder'])
            r_sh = get_landmark(f, POSE_LANDMARKS['right_shoulder'])
            l_el = get_landmark(f, POSE_LANDMARKS['left_elbow'])
            r_el = get_landmark(f, POSE_LANDMARKS['right_elbow'])
            l_wr = get_landmark(f, POSE_LANDMARKS['left_wrist'])
            r_wr = get_landmark(f, POSE_LANDMARKS['right_wrist'])
            l_hip = get_landmark(f, POSE_LANDMARKS['left_hip'])
            r_hip = get_landmark(f, POSE_LANDMARKS['right_hip'])
            nose = get_landmark(f, POSE_LANDMARKS['nose'])
            l_eye = get_landmark(f, POSE_LANDMARKS['left_eye'])
            r_eye = get_landmark(f, POSE_LANDMARKS['right_eye'])
            
            # 팔 움직임 분석
            if l_wr and l_sh and l_wr['y'] < l_sh['y'] - 0.1:
                arms.append('왼팔을 들어올림')
            if r_wr and r_sh and r_wr['y'] < r_sh['y'] - 0.1:
                arms.append('오른팔을 들어올림')
            
            # 손 작업 분석
            if l_wr and l_sh and abs(l_wr['y'] - l_sh['y']) < 0.2:
                hands.append('왼손으로 작업')
            if r_wr and r_sh and abs(r_wr['y'] - r_sh['y']) < 0.2:
                hands.append('오른손으로 작업')
            
            # 몸통 기울기 분석
            if l_sh and r_sh and l_hip and r_hip:
                sh_c = np.array([(l_sh['x']+r_sh['x'])/2, (l_sh['y']+r_sh['y'])/2])
                hip_c = np.array([(l_hip['x']+r_hip['x'])/2, (l_hip['y']+r_hip['y'])/2])
                torso_vec = sh_c - hip_c
                angle = np.arccos(np.clip(np.dot(torso_vec, [0,-1])/np.linalg.norm(torso_vec),-1,1))*180/np.pi
                if angle > 20:
                    torso.append('작업대를 향해 몸을 기울임')
            
            # 머리 방향 분석
            if nose and l_eye and r_eye:
                eye_y = (l_eye['y']+r_eye['y'])/2
                if nose['y'] > eye_y + 0.05:
                    head.append('작업 대상에 집중')
            
            # 정밀작업 감지
            if l_wr and r_wr and l_sh and abs(l_wr['y']-l_sh['y'])<0.2 and abs(r_wr['y']-l_sh['y'])<0.2:
                precision += 1
        
        # 설명 생성
        desc = []
        if hands:
            c = Counter(hands).most_common(1)[0][0]
            desc.append(c)
        if arms:
            c = Counter(arms).most_common(1)[0][0]
            desc.append(c)
        if precision > len(frames)*0.2:
            desc.append('정밀한 조립 작업을 수행')
        if torso:
            c = Counter(torso).most_common(1)[0][0]
            desc.append(c)
        if head:
            c = Counter(head).most_common(1)[0][0]
            desc.append(c)
        if not desc:
            desc.append('작업자가 안정적인 자세를 유지하며 작업을 수행합니다.')
        return ', '.join(desc)
    
    # 텍스트 파일 생성
    seg = int(FPS * SEGMENT_SEC)
    lines = []
    
    for i in range(0, len(frame_data_list), seg):
        s, e = i, min(i+seg, len(frame_data_list))
        t0 = str(timedelta(seconds=int(s/FPS)))[:-3]
        t1 = str(timedelta(seconds=int(e/FPS)))[:-3]
        t0 = t0 if ':' in t0 else '0:00'
        t1 = t1 if ':' in t1 else '0:00'
        desc = analyze_segment(frame_data_list[s:e])
        lines.append(f'**{t0}-{t1}:** {desc}.')
    
    # 텍스트 파일 저장 (text 디렉터리에 생성)
    #txt_path = '../text/worker_activity_description.txt'
    txt_path = '/home/gusdl/deepseek_pj/text/worker_activity_description.txt'
    txt_dir = os.path.dirname(txt_path)
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"작업 설명 텍스트 파일이 '{txt_path}'에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    # 분석할 비디오 파일 경로
    #input_video_file = '../video/sample_video.mp4' 
    input_video_file = '../video/test_topview_one.mp4'

    # 추출된 JSON 파일이 저장될 경로
    output_json_file = '../json/output_skeleton_data.json'

    # 함수 호출
    extract_skeleton_data(input_video_file, output_json_file, visualize=True)
    print("스크립트 실행 완료.") 