import json
import numpy as np
from datetime import datetime, timedelta
import os
from collections import Counter

class FinalWorkAnalyzer:
    def __init__(self, json_file_path, fps=30):
        """
        최종 작업 분석기 초기화
        
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
    
    def analyze_work_sequence(self, start_frame, end_frame):
        """작업 시퀀스를 분석합니다."""
        if not self.frame_data:
            return {}
        
        analysis = {
            'arm_activities': [],
            'hand_activities': [],
            'torso_activities': [],
            'head_activities': [],
            'precision_indicators': 0,
            'work_area_usage': 0,
            'movement_patterns': []
        }
        
        for frame_idx in range(start_frame, min(end_frame, len(self.frame_data))):
            frame_data = self.frame_data[frame_idx]
            
            # 주요 관절 위치 가져오기
            landmarks = {}
            for name, id_num in self.POSE_LANDMARKS.items():
                landmarks[name] = self.get_landmark_position(frame_data, id_num)
            
            # 각 활동 분석
            self.analyze_frame_activities(landmarks, analysis)
        
        return analysis
    
    def analyze_frame_activities(self, landmarks, analysis):
        """단일 프레임의 활동을 분석합니다."""
        # 팔 활동 분석
        left_arm = self.analyze_arm_activity(
            landmarks['left_shoulder'], landmarks['left_elbow'], landmarks['left_wrist'], '왼팔'
        )
        right_arm = self.analyze_arm_activity(
            landmarks['right_shoulder'], landmarks['right_elbow'], landmarks['right_wrist'], '오른팔'
        )
        
        if left_arm:
            analysis['arm_activities'].append(left_arm)
        if right_arm:
            analysis['arm_activities'].append(right_arm)
        
        # 손 활동 분석
        hand_activity = self.analyze_hand_activity(landmarks)
        if hand_activity:
            analysis['hand_activities'].append(hand_activity)
        
        # 몸통 활동 분석
        torso_activity = self.analyze_torso_activity(landmarks)
        if torso_activity:
            analysis['torso_activities'].append(torso_activity)
        
        # 머리 활동 분석
        head_activity = self.analyze_head_activity(landmarks)
        if head_activity:
            analysis['head_activities'].append(head_activity)
        
        # 정밀 작업 지표
        if self.detect_precision_work(landmarks):
            analysis['precision_indicators'] += 1
        
        # 작업 영역 사용
        if self.detect_work_area_usage(landmarks):
            analysis['work_area_usage'] += 1
        
        # 움직임 패턴
        movement = self.detect_movement_pattern(landmarks)
        if movement:
            analysis['movement_patterns'].append(movement)
    
    def analyze_arm_activity(self, shoulder, elbow, wrist, arm_name):
        """팔 활동을 분석합니다."""
        if not all([shoulder, elbow, wrist]) or wrist['visibility'] < 0.5:
            return None
        
        wrist_y = wrist['y']
        shoulder_y = shoulder['y']
        
        # 팔의 높이에 따른 활동 분류
        if wrist_y < shoulder_y - 0.2:
            return f"{arm_name}을 높이 들어올림"
        elif wrist_y < shoulder_y - 0.1:
            return f"{arm_name}을 들어올림"
        elif wrist_y > shoulder_y + 0.2:
            return f"{arm_name}을 내림"
        
        # 팔꿈치 각도 분석
        arm_angle = self.calculate_arm_angle(shoulder, elbow, wrist)
        if 30 < arm_angle < 90:
            return f"{arm_name}으로 정밀 작업 수행"
        
        return None
    
    def analyze_hand_activity(self, landmarks):
        """손 활동을 분석합니다."""
        left_wrist = landmarks['left_wrist']
        right_wrist = landmarks['right_wrist']
        left_shoulder = landmarks['left_shoulder']
        
        if not all([left_wrist, right_wrist, left_shoulder]):
            return None
        
        shoulder_y = left_shoulder['y']
        work_area_min = shoulder_y - 0.25
        work_area_max = shoulder_y + 0.25
        
        left_in_work_area = work_area_min <= left_wrist['y'] <= work_area_max
        right_in_work_area = work_area_min <= right_wrist['y'] <= work_area_max
        
        if left_in_work_area and right_in_work_area:
            return "양손으로 작업 수행"
        elif left_in_work_area:
            return "왼손으로 작업 수행"
        elif right_in_work_area:
            return "오른손으로 작업 수행"
        
        return None
    
    def analyze_torso_activity(self, landmarks):
        """몸통 활동을 분석합니다."""
        if not all([landmarks['left_shoulder'], landmarks['right_shoulder'], 
                   landmarks['left_hip'], landmarks['right_hip']]):
            return None
        
        shoulder_center = np.array([
            (landmarks['left_shoulder']['x'] + landmarks['right_shoulder']['x']) / 2,
            (landmarks['left_shoulder']['y'] + landmarks['right_shoulder']['y']) / 2
        ])
        
        hip_center = np.array([
            (landmarks['left_hip']['x'] + landmarks['right_hip']['x']) / 2,
            (landmarks['left_hip']['y'] + landmarks['right_hip']['y']) / 2
        ])
        
        torso_vector = shoulder_center - hip_center
        vertical_vector = np.array([0, -1])
        
        cos_angle = np.dot(torso_vector, vertical_vector) / np.linalg.norm(torso_vector)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        if angle > 25:
            return "작업대를 향해 몸을 크게 기울임"
        elif angle > 15:
            return "작업대를 향해 몸을 기울임"
        
        return None
    
    def analyze_head_activity(self, landmarks):
        """머리 활동을 분석합니다."""
        if not all([landmarks['nose'], landmarks['left_eye'], landmarks['right_eye']]):
            return None
        
        nose_y = landmarks['nose']['y']
        left_eye_y = landmarks['left_eye']['y']
        right_eye_y = landmarks['right_eye']['y']
        eye_center_y = (left_eye_y + right_eye_y) / 2
        
        if nose_y > eye_center_y + 0.08:
            return "작업 대상에 집중하여 바라봄"
        
        return None
    
    def detect_precision_work(self, landmarks):
        """정밀 작업을 감지합니다."""
        finger_landmarks = [
            landmarks['left_pinky'], landmarks['right_pinky'],
            landmarks['left_index'], landmarks['right_index'],
            landmarks['left_thumb'], landmarks['right_thumb']
        ]
        
        visible_fingers = [f for f in finger_landmarks if f and f['visibility'] > 0.6]
        
        if len(visible_fingers) >= 4:
            distances = []
            for i in range(len(visible_fingers)):
                for j in range(i+1, len(visible_fingers)):
                    dist = self.calculate_distance(visible_fingers[i], visible_fingers[j])
                    distances.append(dist)
            
            if distances and np.mean(distances) < 0.08:
                return True
        
        return False
    
    def detect_work_area_usage(self, landmarks):
        """작업 영역 사용을 감지합니다."""
        left_wrist = landmarks['left_wrist']
        right_wrist = landmarks['right_wrist']
        left_shoulder = landmarks['left_shoulder']
        
        if not all([left_wrist, right_wrist, left_shoulder]):
            return False
        
        shoulder_y = left_shoulder['y']
        work_area_min = shoulder_y - 0.25
        work_area_max = shoulder_y + 0.25
        
        return (work_area_min <= left_wrist['y'] <= work_area_max or 
                work_area_min <= right_wrist['y'] <= work_area_max)
    
    def detect_movement_pattern(self, landmarks):
        """움직임 패턴을 감지합니다."""
        # 간단한 움직임 패턴 감지
        if landmarks['left_wrist'] and landmarks['right_wrist']:
            left_wrist_y = landmarks['left_wrist']['y']
            right_wrist_y = landmarks['right_wrist']['y']
            
            # 양손이 비슷한 높이에 있는지 확인
            if abs(left_wrist_y - right_wrist_y) < 0.1:
                return "양손을 조화롭게 사용"
        
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
        """두 위치 간의 거리를 계산합니다."""
        if not pos1 or not pos2:
            return float('inf')
        
        dx = pos1['x'] - pos2['x']
        dy = pos1['y'] - pos2['y']
        dz = pos1['z'] - pos2['z']
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def generate_work_description(self, segment_duration=7):
        """작업 설명을 생성합니다."""
        if not self.frame_data:
            return "데이터를 로드할 수 없습니다."
        
        total_frames = len(self.frame_data)
        frames_per_segment = int(self.fps * segment_duration)
        descriptions = []
        
        print(f"전체 {total_frames} 프레임을 {segment_duration}초씩 분석 중...")
        
        for segment_start in range(0, total_frames, frames_per_segment):
            segment_end = min(segment_start + frames_per_segment, total_frames)
            
            # 시간 계산
            start_time = segment_start / self.fps
            end_time = segment_end / self.fps
            
            # 시간 형식 변환
            start_str = str(timedelta(seconds=int(start_time)))
            end_str = str(timedelta(seconds=int(end_time)))
            
            # 해당 구간의 분석
            analysis = self.analyze_work_sequence(segment_start, segment_end)
            
            # 자연스러운 설명 생성
            description = self.create_work_description(analysis, start_str, end_str)
            descriptions.append(description)
        
        return "\n".join(descriptions)
    
    def create_work_description(self, analysis, start_str, end_str):
        """분석 결과를 기반으로 작업 설명을 생성합니다."""
        # 주요 활동 파악
        arm_activities = Counter(analysis['arm_activities'])
        hand_activities = Counter(analysis['hand_activities'])
        torso_activities = Counter(analysis['torso_activities'])
        head_activities = Counter(analysis['head_activities'])
        
        # 가장 빈번한 활동들
        main_arm_activity = arm_activities.most_common(1)[0][0] if arm_activities else None
        main_hand_activity = hand_activities.most_common(1)[0][0] if hand_activities else None
        main_torso_activity = torso_activities.most_common(1)[0][0] if torso_activities else None
        main_head_activity = head_activities.most_common(1)[0][0] if head_activities else None
        
        # 비율 계산
        total_frames = len(analysis['arm_activities']) + len(analysis['hand_activities'])
        precision_ratio = analysis['precision_indicators'] / total_frames if total_frames > 0 else 0
        work_area_ratio = analysis['work_area_usage'] / total_frames if total_frames > 0 else 0
        
        # 작업 유형 결정
        work_type = self.determine_work_type(analysis)
        
        # 설명 구성
        description_parts = []
        
        # 기본 작업 활동
        if main_hand_activity:
            description_parts.append(main_hand_activity)
        
        if main_arm_activity:
            description_parts.append(main_arm_activity)
        
        # 정밀 작업 언급
        if precision_ratio > 0.25:
            description_parts.append("정밀한 조립 작업을 수행")
        
        # 작업 영역 활동
        if work_area_ratio > 0.4:
            description_parts.append("작업대에서 체계적으로 작업")
        
        # 몸 자세
        if main_torso_activity:
            description_parts.append(main_torso_activity)
        
        if main_head_activity:
            description_parts.append(main_head_activity)
        
        # 설명 조합
        if description_parts:
            description = f"**{start_str}-{end_str}:** " + ", ".join(description_parts) + "."
        else:
            description = f"**{start_str}-{end_str}:** 작업자가 안정적인 자세를 유지하며 {work_type}을 수행합니다."
        
        return description
    
    def determine_work_type(self, analysis):
        """분석 결과를 기반으로 작업 유형을 결정합니다."""
        # 정밀 작업 비율
        total_frames = len(analysis['arm_activities']) + len(analysis['hand_activities'])
        precision_ratio = analysis['precision_indicators'] / total_frames if total_frames > 0 else 0
        
        # 작업 영역 사용 비율
        work_area_ratio = analysis['work_area_usage'] / total_frames if total_frames > 0 else 0
        
        # 팔 활동 분석
        arm_text = " ".join(analysis['arm_activities']).lower()
        
        if precision_ratio > 0.3:
            return "정밀 조립 작업"
        elif "들어올림" in arm_text or "내림" in arm_text:
            return "부품 조작 작업"
        elif work_area_ratio > 0.5:
            return "조립 작업"
        else:
            return "일반 작업"
    
    def save_work_description(self, output_file_path, segment_duration=7):
        """작업 설명을 텍스트 파일로 저장합니다."""
        description = self.generate_work_description(segment_duration)
        
        # 출력 디렉터리 생성
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 파일 저장
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("작업자 활동 분석 결과\n")
            f.write("=" * 50 + "\n\n")
            f.write("이 분석은 MediaPipe를 통해 추출된 스켈레톤 데이터를 기반으로\n")
            f.write("작업자의 활동을 시간대별로 분석하여 자연스러운 한국어로 설명합니다.\n\n")
            f.write("분석 기준:\n")
            f.write("- 팔의 움직임과 위치\n")
            f.write("- 손의 작업 영역 사용\n")
            f.write("- 몸통의 기울기\n")
            f.write("- 머리의 방향\n")
            f.write("- 정밀 작업 패턴\n\n")
            f.write("=" * 50 + "\n\n")
            f.write(description)
        
        print(f"작업 설명이 '{output_file_path}'에 저장되었습니다.")
        return description

def main():
    """메인 함수"""
    # JSON 파일 경로 (tf_script.py에서 생성된 파일)
    json_file_path = '../json/output_skeleton_data.json'
    
    # 분석기 생성
    analyzer = FinalWorkAnalyzer(json_file_path, fps=30)
    
    # 결과 파일 경로
    output_file_path = '../analysis/final_work_description.txt'
    
    # 작업 설명 생성 및 저장
    description = analyzer.save_work_description(output_file_path, segment_duration=7)
    
    # 콘솔에도 출력
    print("\n" + "=" * 50)
    print("작업자 활동 분석 결과")
    print("=" * 50)
    print(description)

if __name__ == "__main__":
    main() 