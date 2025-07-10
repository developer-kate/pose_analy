import cv2
import mediapipe as mp
import json
import os
import numpy as np
from collections import Counter
from datetime import timedelta

def extract_skeleton_data(video_path, output_json_path, visualize=False):
    # ... 기존 코드 생략 ...
    # JSON 파일로 저장
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(frame_data_list, f, ensure_ascii=False, indent=4)
    print(f"스켈레톤 데이터가 '{output_json_path}'에 성공적으로 저장되었습니다.")
    generate_work_description(frame_data_list)

def generate_work_description(frame_data_list):
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
            if l_wr and l_sh and l_wr['y'] < l_sh['y'] - 0.1:
                arms.append('왼팔을 들어올림')
            if r_wr and r_sh and r_wr['y'] < r_sh['y'] - 0.1:
                arms.append('오른팔을 들어올림')
            if l_wr and l_sh and abs(l_wr['y'] - l_sh['y']) < 0.2:
                hands.append('왼손으로 작업')
            if r_wr and r_sh and abs(r_wr['y'] - r_sh['y']) < 0.2:
                hands.append('오른손으로 작업')
            if l_sh and r_sh and l_hip and r_hip:
                sh_c = np.array([(l_sh['x']+r_sh['x'])/2, (l_sh['y']+r_sh['y'])/2])
                hip_c = np.array([(l_hip['x']+r_hip['x'])/2, (l_hip['y']+r_hip['y'])/2])
                torso_vec = sh_c - hip_c
                angle = np.arccos(np.clip(np.dot(torso_vec, [0,-1])/np.linalg.norm(torso_vec),-1,1))*180/np.pi
                if angle > 20:
                    torso.append('작업대를 향해 몸을 기울임')
            if nose and l_eye and r_eye:
                eye_y = (l_eye['y']+r_eye['y'])/2
                if nose['y'] > eye_y + 0.05:
                    head.append('작업 대상에 집중')
            if l_wr and r_wr and l_sh and abs(l_wr['y']-l_sh['y'])<0.2 and abs(r_wr['y']-l_sh['y'])<0.2:
                precision += 1
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
    # 텍스트 파일 저장 (절대경로)
    txt_path = '/home/gusdl/deepseek_pj/text/worker_activity_description.txt'
    txt_dir = os.path.dirname(txt_path)
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"작업 설명 텍스트 파일이 '{txt_path}'에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    input_video_file = '../video/test_topview_one.mp4'
    output_json_file = '../json/output_skeleton_data.json'
    extract_skeleton_data(input_video_file, output_json_file, visualize=True)
    print("스크립트 실행 완료.") 