import json
import numpy as np
from datetime import datetime, timedelta
import os
from collections import Counter

class NaturalWorkDescription:
    def __init__(self, json_file_path, fps=30):
        """
        자연스러운 작업 설명 생성기 초기화
        
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
    
    def analyze_work_pattern(self, start_frame, end_frame):
        """작업 패턴을 분석합니다."""
        if not self.frame_data:
            return {}
        
        patterns = {
            'arm_movements': [],
            'hand_positions': [],
            'torso_movements': [],
            'head_movements': [],
            'precision_work': 0,
            'work_area_activity': 0
        }
        
        for frame_idx in range(start_frame, min(end_frame, len(self.frame_data))):
            frame_data = self.frame_data[frame_idx]
            
            # 주요 관절 위치 가져오기
            landmarks = {}
            for name, id_num in self.POSE_LANDMARKS.items():
                landmarks[name] = self.get_landmark_position(frame_data, id_num)
            
            # 각 패턴 분석
            self.analyze_frame_patterns(landmarks, patterns)
        
        return patterns
    
    def analyze_frame_patterns(self, landmarks, patterns):
        """단일 프레임의 패턴을 분석합니다."""
        # 팔 움직임 분석
        left_arm = self.analyze_arm_movement(
            landmarks['left_shoulder'], landmarks['left_elbow'], landmarks['left_wrist']
        )
        right_arm = self.analyze_arm_movement(
            landmarks['right_shoulder'], landmarks['right_elbow'], landmarks['right_wrist']
        )
        
        if left_arm:
            patterns['arm_movements'].append(left_arm)
        if right_arm:
            patterns['arm_movements'].append(right_arm)
        
        # 손 위치 분석
        hand_pos = self.analyze_hand_position(landmarks)
        if hand_pos:
            patterns['hand_positions'].append(hand_pos)
        
        # 몸통 움직임 분석
        torso = self.analyze_torso_movement(landmarks)
        if torso:
            patterns['torso_movements'].append(torso)
        
        # 머리 움직임 분석
        head = self.analyze_head_movement(landmarks)
        if head:
            patterns['head_movements'].append(head)
        
        # 정밀 작업 감지
        if self.detect_precision_work(landmarks):
            patterns['precision_work'] += 1
        
        # 작업 영역 활동 감지
        if self.detect_work_area_activity(landmarks):
            patterns['work_area_activity'] += 1
    
    def analyze_arm_movement(self, shoulder, elbow, wrist):
        """팔 움직임을 분석합니다."""
        if not all([shoulder, elbow, wrist]) or wrist['visibility'] < 0.5:
            return None
        
        wrist_y = wrist['y']
        shoulder_y = shoulder['y']
        
        if wrist_y < shoulder_y - 0.15:
            return "팔을 높이 들어올림"
        elif wrist_y < shoulder_y - 0.05:
            return "팔을 들어올림"
        elif wrist_y > shoulder_y + 0.15:
            return "팔을 내림"
        
        return None
    
    def analyze_hand_position(self, landmarks):
        """손 위치를 분석합니다."""
        left_wrist = landmarks['left_wrist']
        right_wrist = landmarks['right_wrist']
        left_shoulder = landmarks['left_shoulder']
        
        if not all([left_wrist, right_wrist, left_shoulder]):
            return None
        
        shoulder_y = left_shoulder['y']
        work_area_min = shoulder_y - 0.2
        work_area_max = shoulder_y + 0.2
        
        left_in_work_area = work_area_min <= left_wrist['y'] <= work_area_max
        right_in_work_area = work_area_min <= right_wrist['y'] <= work_area_max
        
        if left_in_work_area and right_in_work_area:
            return "양손으로 작업 수행"
        elif left_in_work_area:
            return "왼손으로 작업 수행"
        elif right_in_work_area:
            return "오른손으로 작업 수행"
        
        return None
    
    def analyze_torso_movement(self, landmarks):
        """몸통 움직임을 분석합니다."""
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
        
        if angle > 20:
            return "몸을 크게 기울임"
        elif angle > 10:
            return "몸을 기울임"
        
        return None
    
    def analyze_head_movement(self, landmarks):
        """머리 움직임을 분석합니다."""
        if not all([landmarks['nose'], landmarks['left_eye'], landmarks['right_eye']]):
            return None
        
        nose_y = landmarks['nose']['y']
        left_eye_y = landmarks['left_eye']['y']
        right_eye_y = landmarks['right_eye']['y']
        eye_center_y = (left_eye_y + right_eye_y) / 2
        
        if nose_y > eye_center_y + 0.05:
            return "아래쪽을 바라봄"
        
        return None
    
    def detect_precision_work(self, landmarks):
        """정밀 작업을 감지합니다."""
        finger_landmarks = [
            landmarks['left_pinky'], landmarks['right_pinky'],
            landmarks['left_index'], landmarks['right_index'],
            landmarks['left_thumb'], landmarks['right_thumb']
        ]
        
        visible_fingers = [f for f in finger_landmarks if f and f['visibility'] > 0.5]
        
        if len(visible_fingers) >= 4:
            distances = []
            for i in range(len(visible_fingers)):
                for j in range(i+1, len(visible_fingers)):
                    dist = self.calculate_distance(visible_fingers[i], visible_fingers[j])
                    distances.append(dist)
            
            if distances and np.mean(distances) < 0.1:
                return True
        
        return False
    
    def detect_work_area_activity(self, landmarks):
        """작업 영역 활동을 감지합니다."""
        left_wrist = landmarks['left_wrist']
        right_wrist = landmarks['right_wrist']
        left_shoulder = landmarks['left_shoulder']
        
        if not all([left_wrist, right_wrist, left_shoulder]):
            return False
        
        shoulder_y = left_shoulder['y']
        work_area_min = shoulder_y - 0.2
        work_area_max = shoulder_y + 0.2
        
        return (work_area_min <= left_wrist['y'] <= work_area_max or 
                work_area_min <= right_wrist['y'] <= work_area_max)
    
    def calculate_distance(self, pos1, pos2):
        """두 위치 간의 거리를 계산합니다."""
        if not pos1 or not pos2:
            return float('inf')
        
        dx = pos1['x'] - pos2['x']
        dy = pos1['y'] - pos2['y']
        dz = pos1['z'] - pos2['z']
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def generate_natural_description(self, segment_duration=7):
        """자연스러운 작업 설명을 생성합니다."""
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
            
            # 해당 구간의 패턴 분석
            patterns = self.analyze_work_pattern(segment_start, segment_end)
            
            # 자연스러운 설명 생성
            description = self.create_natural_description(patterns, start_str, end_str)
            descriptions.append(description)
        
        return "\n".join(descriptions)
    
    def create_natural_description(self, patterns, start_str, end_str):
        """패턴을 기반으로 자연스러운 설명을 생성합니다."""
        # 주요 활동 파악
        arm_movements = Counter(patterns['arm_movements'])
        hand_positions = Counter(patterns['hand_positions'])
        torso_movements = Counter(patterns['torso_movements'])
        head_movements = Counter(patterns['head_movements'])
        
        # 가장 빈번한 활동들
        main_arm_movement = arm_movements.most_common(1)[0][0] if arm_movements else None
        main_hand_position = hand_positions.most_common(1)[0][0] if hand_positions else None
        main_torso_movement = torso_movements.most_common(1)[0][0] if torso_movements else None
        main_head_movement = head_movements.most_common(1)[0][0] if head_movements else None
        
        # 정밀 작업 비율
        total_frames = sum(patterns.values())
        precision_ratio = patterns['precision_work'] / total_frames if total_frames > 0 else 0
        work_area_ratio = patterns['work_area_activity'] / total_frames if total_frames > 0 else 0
        
        # 자연스러운 설명 생성
        description_parts = []
        
        # 기본 작업 활동
        if main_hand_position:
            description_parts.append(main_hand_position)
        
        if main_arm_movement:
            description_parts.append(main_arm_movement)
        
        # 정밀 작업 언급
        if precision_ratio > 0.3:
            description_parts.append("정밀한 조립 작업을 수행")
        
        # 작업 영역 활동
        if work_area_ratio > 0.5:
            description_parts.append("작업대에서 체계적으로 작업")
        
        # 몸 자세
        if main_torso_movement:
            description_parts.append(main_torso_movement)
        
        if main_head_movement:
            description_parts.append("작업 대상에 집중")
        
        # 설명 조합
        if description_parts:
            description = f"**{start_str}-{end_str}:** " + ", ".join(description_parts) + "."
        else:
            description = f"**{start_str}-{end_str}:** 작업자가 안정적인 자세를 유지하며 작업을 수행합니다."
        
        return description
    
    def save_natural_description(self, output_file_path, segment_duration=7):
        """자연스러운 설명을 텍스트 파일로 저장합니다."""
        description = self.generate_natural_description(segment_duration)
        
        # 출력 디렉터리 생성
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 파일 저장
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("작업자 활동 자연스러운 설명\n")
            f.write("=" * 50 + "\n\n")
            f.write("이 분석은 MediaPipe를 통해 추출된 스켈레톤 데이터를 기반으로\n")
            f.write("작업자의 활동을 자연스러운 한국어로 설명합니다.\n\n")
            f.write("=" * 50 + "\n\n")
            f.write(description)
        
        print(f"자연스러운 설명이 '{output_file_path}'에 저장되었습니다.")
        return description

def main():
    """메인 함수"""
    # JSON 파일 경로 (tf_script.py에서 생성된 파일)
    json_file_path = '../json/output_skeleton_data.json'
    
    # 설명 생성기 생성
    describer = NaturalWorkDescription(json_file_path, fps=30)
    
    # 결과 파일 경로
    output_file_path = '../analysis/natural_work_description.txt'
    
    # 자연스러운 설명 생성 및 저장
    description = describer.save_natural_description(output_file_path, segment_duration=7)
    
    # 콘솔에도 출력
    print("\n" + "=" * 50)
    print("작업자 활동 자연스러운 설명")
    print("=" * 50)
    print(description)

if __name__ == "__main__":
    main() 