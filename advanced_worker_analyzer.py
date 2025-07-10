import json
import numpy as np
from datetime import datetime, timedelta
import os
from collections import defaultdict, Counter

class AdvancedWorkerAnalyzer:
    def __init__(self, json_file_path, fps=30):
        """
        고급 작업자 활동 분석기 초기화
        
        Args:
            json_file_path (str): 스켈레톤 데이터가 저장된 JSON 파일 경로
            fps (int): 비디오의 프레임 레이트 (기본값: 30)
        """
        self.json_file_path = json_file_path
        self.fps = fps
        self.frame_data = None
        self.load_data()
        
        # MediaPipe Pose 랜드마크 ID 정의
        self.POSE_LANDMARKS = {
            'nose': 0,
            'left_eye': 2,
            'right_eye': 5,
            'left_ear': 7,
            'right_ear': 8,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_pinky': 17,
            'right_pinky': 18,
            'left_index': 19,
            'right_index': 20,
            'left_thumb': 21,
            'right_thumb': 22,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
    def load_data(self):
        """JSON 파일에서 스켈레톤 데이터를 로드합니다."""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.frame_data = json.load(f)
            print(f"데이터 로드 완료: {len(self.frame_data)} 프레임")
        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다. {self.json_file_path}")
            return
        except json.JSONDecodeError:
            print(f"오류: JSON 파일 형식이 올바르지 않습니다. {self.json_file_path}")
            return
    
    def get_landmark_position(self, frame_data, landmark_id):
        """특정 랜드마크의 위치를 반환합니다."""
        if not frame_data.get('landmarks'):
            return None
        
        for landmark in frame_data['landmarks']:
            if landmark['id'] == landmark_id:
                return {
                    'x': landmark['x'],
                    'y': landmark['y'],
                    'z': landmark['z'],
                    'visibility': landmark['visibility']
                }
        return None
    
    def analyze_work_activity(self, start_frame, end_frame):
        """작업 활동을 분석합니다."""
        if not self.frame_data:
            return []
        
        activities = []
        
        for frame_idx in range(start_frame, min(end_frame, len(self.frame_data))):
            frame_data = self.frame_data[frame_idx]
            
            # 주요 관절 위치 가져오기
            landmarks = {}
            for name, id_num in self.POSE_LANDMARKS.items():
                landmarks[name] = self.get_landmark_position(frame_data, id_num)
            
            # 활동 분석
            frame_activities = self.analyze_single_frame_work_activity(landmarks)
            if frame_activities:
                activities.extend(frame_activities)
        
        return activities
    
    def analyze_single_frame_work_activity(self, landmarks):
        """단일 프레임의 작업 활동을 분석합니다."""
        activities = []
        
        # 1. 팔의 높이와 위치 분석
        left_arm_activity = self.analyze_arm_position(
            landmarks['left_shoulder'], landmarks['left_elbow'], landmarks['left_wrist'], '왼팔'
        )
        right_arm_activity = self.analyze_arm_position(
            landmarks['right_shoulder'], landmarks['right_elbow'], landmarks['right_wrist'], '오른팔'
        )
        
        if left_arm_activity:
            activities.append(left_arm_activity)
        if right_arm_activity:
            activities.append(right_arm_activity)
        
        # 2. 정밀 작업 감지 (손가락 관절 분석)
        precision_work = self.detect_precision_work(landmarks)
        if precision_work:
            activities.append(precision_work)
        
        # 3. 몸통 기울기 분석
        torso_activity = self.analyze_torso_movement(landmarks)
        if torso_activity:
            activities.append(torso_activity)
        
        # 4. 머리 방향 분석 (작업 대상 확인)
        head_activity = self.analyze_head_movement(landmarks)
        if head_activity:
            activities.append(head_activity)
        
        # 5. 작업 영역 분석
        work_area_activity = self.analyze_work_area(landmarks)
        if work_area_activity:
            activities.append(work_area_activity)
        
        return activities
    
    def analyze_arm_position(self, shoulder, elbow, wrist, arm_name):
        """팔의 위치와 자세를 분석합니다."""
        if not all([shoulder, elbow, wrist]) or wrist['visibility'] < 0.5:
            return None
        
        # 팔의 높이 분석
        wrist_height = wrist['y']
        shoulder_height = shoulder['y']
        
        if wrist_height < shoulder_height - 0.15:  # 손목이 어깨보다 훨씬 높음
            return f"{arm_name}을 높이 들어올림"
        elif wrist_height < shoulder_height - 0.05:  # 손목이 어깨보다 약간 높음
            return f"{arm_name}을 들어올림"
        elif wrist_height > shoulder_height + 0.15:  # 손목이 어깨보다 훨씬 낮음
            return f"{arm_name}을 내림"
        
        # 팔꿈치 각도 분석
        arm_angle = self.calculate_arm_angle(shoulder, elbow, wrist)
        if 45 < arm_angle < 90:  # 팔꿈치가 적당히 구부러짐
            return f"{arm_name}으로 정밀 작업 수행"
        elif arm_angle < 45:  # 팔꿈치가 많이 구부러짐
            return f"{arm_name}으로 세밀한 작업 수행"
        
        return None
    
    def detect_precision_work(self, landmarks):
        """정밀 작업을 감지합니다."""
        # 손가락 관절들의 위치를 확인
        finger_landmarks = [
            landmarks['left_pinky'], landmarks['right_pinky'],
            landmarks['left_index'], landmarks['right_index'],
            landmarks['left_thumb'], landmarks['right_thumb']
        ]
        
        # 손가락들이 모여있는지 확인 (정밀 작업의 지표)
        visible_fingers = [f for f in finger_landmarks if f and f['visibility'] > 0.5]
        
        if len(visible_fingers) >= 4:  # 최소 4개의 손가락이 보임
            # 손가락들 간의 거리 계산
            distances = []
            for i in range(len(visible_fingers)):
                for j in range(i+1, len(visible_fingers)):
                    dist = self.calculate_distance(visible_fingers[i], visible_fingers[j])
                    distances.append(dist)
            
            if distances and np.mean(distances) < 0.1:  # 손가락들이 가까이 모여있음
                return "정밀한 손 작업 수행"
        
        return None
    
    def analyze_torso_movement(self, landmarks):
        """몸통의 움직임을 분석합니다."""
        if not all([landmarks['left_shoulder'], landmarks['right_shoulder'], 
                   landmarks['left_hip'], landmarks['right_hip']]):
            return None
        
        # 어깨와 엉덩이의 기울기 계산
        shoulder_center = np.array([
            (landmarks['left_shoulder']['x'] + landmarks['right_shoulder']['x']) / 2,
            (landmarks['left_shoulder']['y'] + landmarks['right_shoulder']['y']) / 2
        ])
        
        hip_center = np.array([
            (landmarks['left_hip']['x'] + landmarks['right_hip']['x']) / 2,
            (landmarks['left_hip']['y'] + landmarks['right_hip']['y']) / 2
        ])
        
        # 수직 벡터와의 각도 계산
        torso_vector = shoulder_center - hip_center
        vertical_vector = np.array([0, -1])
        
        cos_angle = np.dot(torso_vector, vertical_vector) / np.linalg.norm(torso_vector)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        if angle > 20:  # 몸통이 많이 기울어짐
            return "작업대를 향해 몸을 크게 기울임"
        elif angle > 10:  # 몸통이 약간 기울어짐
            return "작업대를 향해 몸을 기울임"
        
        return None
    
    def analyze_head_movement(self, landmarks):
        """머리의 움직임을 분석합니다."""
        if not all([landmarks['nose'], landmarks['left_eye'], landmarks['right_eye']]):
            return None
        
        # 머리가 아래쪽을 향하는지 확인 (작업 대상 확인)
        nose_y = landmarks['nose']['y']
        left_eye_y = landmarks['left_eye']['y']
        right_eye_y = landmarks['right_eye']['y']
        
        eye_center_y = (left_eye_y + right_eye_y) / 2
        
        if nose_y > eye_center_y + 0.05:  # 코가 눈보다 아래쪽에 있음
            return "작업 대상에 집중하여 바라봄"
        
        return None
    
    def analyze_work_area(self, landmarks):
        """작업 영역을 분석합니다."""
        if not all([landmarks['left_wrist'], landmarks['right_wrist'], 
                   landmarks['left_shoulder'], landmarks['right_shoulder']]):
            return None
        
        # 양손이 어깨 높이 근처에 있는지 확인 (작업대 영역)
        left_wrist_y = landmarks['left_wrist']['y']
        right_wrist_y = landmarks['right_wrist']['y']
        shoulder_y = landmarks['left_shoulder']['y']
        
        # 작업대 영역 정의 (어깨 높이 ± 0.2)
        work_area_y_min = shoulder_y - 0.2
        work_area_y_max = shoulder_y + 0.2
        
        if (work_area_y_min <= left_wrist_y <= work_area_y_max and 
            work_area_y_min <= right_wrist_y <= work_area_y_max):
            return "작업대에서 양손으로 작업 수행"
        elif work_area_y_min <= left_wrist_y <= work_area_y_max:
            return "작업대에서 왼손으로 작업 수행"
        elif work_area_y_min <= right_wrist_y <= work_area_y_max:
            return "작업대에서 오른손으로 작업 수행"
        
        return None
    
    def calculate_arm_angle(self, shoulder, elbow, wrist):
        """팔꿈치 각도를 계산합니다."""
        if not all([shoulder, elbow, wrist]):
            return 0
        
        v1 = np.array([shoulder['x'] - elbow['x'], shoulder['y'] - elbow['y']])
        v2 = np.array([wrist['x'] - elbow['x'], wrist['y'] - elbow['y']])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def calculate_distance(self, pos1, pos2):
        """두 위치 간의 유클리드 거리를 계산합니다."""
        if not pos1 or not pos2:
            return float('inf')
        
        dx = pos1['x'] - pos2['x']
        dy = pos1['y'] - pos2['y']
        dz = pos1['z'] - pos2['z']
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def generate_detailed_work_summary(self, segment_duration=7):
        """상세한 작업 활동 요약을 생성합니다."""
        if not self.frame_data:
            return "데이터를 로드할 수 없습니다."
        
        total_frames = len(self.frame_data)
        frames_per_segment = int(self.fps * segment_duration)
        summary = []
        
        print(f"전체 {total_frames} 프레임을 {segment_duration}초씩 분석 중...")
        
        for segment_start in range(0, total_frames, frames_per_segment):
            segment_end = min(segment_start + frames_per_segment, total_frames)
            
            # 시간 계산
            start_time = segment_start / self.fps
            end_time = segment_end / self.fps
            
            # 시간 형식 변환
            start_str = str(timedelta(seconds=int(start_time)))
            end_str = str(timedelta(seconds=int(end_time)))
            
            # 해당 구간의 활동 분석
            activities = self.analyze_work_activity(segment_start, segment_end)
            
            # 활동 빈도 계산
            activity_counts = Counter(activities)
            
            # 주요 활동 결정 (20% 이상의 프레임에서 발생)
            threshold = len(activities) * 0.2 if activities else 0
            main_activities = [activity for activity, count in activity_counts.items() 
                             if count >= threshold]
            
            # 작업 유형 분류
            work_type = self.classify_work_type(activities)
            
            # 요약 텍스트 생성
            if main_activities:
                activity_text = ", ".join(main_activities)
                summary.append(f"**{start_str}-{end_str}:** {work_type} {activity_text}")
            else:
                summary.append(f"**{start_str}-{end_str}:** 작업자가 안정적인 자세를 유지하며 {work_type}을 수행합니다.")
        
        return "\n".join(summary)
    
    def classify_work_type(self, activities):
        """활동을 기반으로 작업 유형을 분류합니다."""
        activity_text = " ".join(activities).lower()
        
        if "정밀" in activity_text or "세밀" in activity_text:
            return "정밀 조립 작업"
        elif "부품" in activity_text or "조립" in activity_text:
            return "조립 작업"
        elif "확인" in activity_text or "검사" in activity_text:
            return "품질 검사 작업"
        elif "선택" in activity_text or "찾" in activity_text:
            return "부품 선택 작업"
        else:
            return "일반 작업"
    
    def save_detailed_analysis(self, output_file_path, segment_duration=7):
        """상세한 분석 결과를 텍스트 파일로 저장합니다."""
        summary = self.generate_detailed_work_summary(segment_duration)
        
        # 출력 디렉터리 생성
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 파일 저장
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("고급 작업자 활동 분석 결과\n")
            f.write("=" * 60 + "\n\n")
            f.write("분석 기준:\n")
            f.write("- 팔의 높이와 각도\n")
            f.write("- 손가락의 정밀 작업 패턴\n")
            f.write("- 몸통의 기울기\n")
            f.write("- 머리의 방향\n")
            f.write("- 작업 영역에서의 활동\n\n")
            f.write("=" * 60 + "\n\n")
            f.write(summary)
        
        print(f"상세 분석 결과가 '{output_file_path}'에 저장되었습니다.")
        return summary

def main():
    """메인 함수"""
    # JSON 파일 경로 (tf_script.py에서 생성된 파일)
    json_file_path = '../json/output_skeleton_data.json'
    
    # 분석기 생성
    analyzer = AdvancedWorkerAnalyzer(json_file_path, fps=30)
    
    # 분석 결과 파일 경로
    output_file_path = '../analysis/advanced_worker_analysis.txt'
    
    # 분석 실행 및 저장
    summary = analyzer.save_detailed_analysis(output_file_path, segment_duration=7)
    
    # 콘솔에도 출력
    print("\n" + "=" * 60)
    print("고급 작업자 활동 분석 결과")
    print("=" * 60)
    print(summary)

if __name__ == "__main__":
    main() 