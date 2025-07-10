import json
import numpy as np
from datetime import datetime, timedelta
import os

class SkeletonAnalyzer:
    def __init__(self, json_file_path, fps=30):
        """
        스켈레톤 데이터 분석기 초기화
        
        Args:
            json_file_path (str): 스켈레톤 데이터가 저장된 JSON 파일 경로
            fps (int): 비디오의 프레임 레이트 (기본값: 30)
        """
        self.json_file_path = json_file_path
        self.fps = fps
        self.frame_data = None
        self.load_data()
        
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
    
    def calculate_distance(self, pos1, pos2):
        """두 위치 간의 유클리드 거리를 계산합니다."""
        if not pos1 or not pos2:
            return float('inf')
        
        dx = pos1['x'] - pos2['x']
        dy = pos1['y'] - pos2['y']
        dz = pos1['z'] - pos2['z']
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def analyze_pose_activity(self, start_frame, end_frame):
        """특정 프레임 범위에서 포즈 활동을 분석합니다."""
        if not self.frame_data:
            return "데이터를 로드할 수 없습니다."
        
        # MediaPipe Pose 랜드마크 ID 정의
        POSE_LANDMARKS = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
        activities = []
        
        for frame_idx in range(start_frame, min(end_frame, len(self.frame_data))):
            frame_data = self.frame_data[frame_idx]
            
            # 주요 관절 위치 가져오기
            left_shoulder = self.get_landmark_position(frame_data, POSE_LANDMARKS['left_shoulder'])
            right_shoulder = self.get_landmark_position(frame_data, POSE_LANDMARKS['right_shoulder'])
            left_elbow = self.get_landmark_position(frame_data, POSE_LANDMARKS['left_elbow'])
            right_elbow = self.get_landmark_position(frame_data, POSE_LANDMARKS['right_elbow'])
            left_wrist = self.get_landmark_position(frame_data, POSE_LANDMARKS['left_wrist'])
            right_wrist = self.get_landmark_position(frame_data, POSE_LANDMARKS['right_wrist'])
            left_hip = self.get_landmark_position(frame_data, POSE_LANDMARKS['left_hip'])
            right_hip = self.get_landmark_position(frame_data, POSE_LANDMARKS['right_hip'])
            
            # 활동 분석
            activity = self.analyze_single_frame_activity(
                left_shoulder, right_shoulder, left_elbow, right_elbow,
                left_wrist, right_wrist, left_hip, right_hip
            )
            
            if activity:
                activities.append(activity)
        
        return activities
    
    def analyze_single_frame_activity(self, left_shoulder, right_shoulder, left_elbow, 
                                    right_elbow, left_wrist, right_wrist, left_hip, right_hip):
        """단일 프레임의 활동을 분석합니다."""
        activities = []
        
        # 팔의 높이 분석 (어깨 대비 손목 위치)
        if left_wrist and left_shoulder and left_wrist['visibility'] > 0.5:
            if left_wrist['y'] < left_shoulder['y'] - 0.1:  # 손목이 어깨보다 높음
                activities.append("왼팔을 들어올림")
            elif left_wrist['y'] > left_shoulder['y'] + 0.1:  # 손목이 어깨보다 낮음
                activities.append("왼팔을 내림")
        
        if right_wrist and right_shoulder and right_wrist['visibility'] > 0.5:
            if right_wrist['y'] < right_shoulder['y'] - 0.1:
                activities.append("오른팔을 들어올림")
            elif right_wrist['y'] > right_shoulder['y'] + 0.1:
                activities.append("오른팔을 내림")
        
        # 팔꿈치 각도 분석 (조립 작업 감지)
        if left_elbow and left_shoulder and left_wrist:
            left_arm_angle = self.calculate_arm_angle(left_shoulder, left_elbow, left_wrist)
            if 60 < left_arm_angle < 120:  # 팔꿈치가 구부러진 상태
                activities.append("왼팔로 정밀 작업 수행")
        
        if right_elbow and right_shoulder and right_wrist:
            right_arm_angle = self.calculate_arm_angle(right_shoulder, right_elbow, right_wrist)
            if 60 < right_arm_angle < 120:
                activities.append("오른팔로 정밀 작업 수행")
        
        # 몸통 기울기 분석
        if left_shoulder and right_shoulder and left_hip and right_hip:
            torso_angle = self.calculate_torso_angle(left_shoulder, right_shoulder, left_hip, right_hip)
            if abs(torso_angle) > 15:  # 몸통이 기울어짐
                activities.append("작업대를 향해 몸을 기울임")
        
        return activities
    
    def calculate_arm_angle(self, shoulder, elbow, wrist):
        """팔꿈치 각도를 계산합니다."""
        if not all([shoulder, elbow, wrist]):
            return 0
        
        # 벡터 계산
        v1 = np.array([shoulder['x'] - elbow['x'], shoulder['y'] - elbow['y']])
        v2 = np.array([wrist['x'] - elbow['x'], wrist['y'] - elbow['y']])
        
        # 각도 계산
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def calculate_torso_angle(self, left_shoulder, right_shoulder, left_hip, right_hip):
        """몸통 기울기 각도를 계산합니다."""
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return 0
        
        # 어깨와 엉덩이의 중점 계산
        shoulder_center = np.array([
            (left_shoulder['x'] + right_shoulder['x']) / 2,
            (left_shoulder['y'] + right_shoulder['y']) / 2
        ])
        
        hip_center = np.array([
            (left_hip['x'] + right_hip['x']) / 2,
            (left_hip['y'] + right_hip['y']) / 2
        ])
        
        # 수직 벡터와의 각도 계산
        torso_vector = shoulder_center - hip_center
        vertical_vector = np.array([0, -1])  # 위쪽 방향
        
        cos_angle = np.dot(torso_vector, vertical_vector) / np.linalg.norm(torso_vector)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def generate_activity_summary(self, segment_duration=7):
        """전체 비디오의 활동 요약을 생성합니다."""
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
            activities = self.analyze_pose_activity(segment_start, segment_end)
            
            # 활동 빈도 계산
            activity_counts = {}
            for activity_list in activities:
                for activity in activity_list:
                    activity_counts[activity] = activity_counts.get(activity, 0) + 1
            
            # 주요 활동 결정
            main_activities = []
            for activity, count in activity_counts.items():
                if count > len(activities) * 0.3:  # 30% 이상의 프레임에서 발생
                    main_activities.append(activity)
            
            # 요약 텍스트 생성
            if main_activities:
                activity_text = ", ".join(main_activities)
                summary.append(f"**{start_str}-{end_str}:** {activity_text}")
            else:
                summary.append(f"**{start_str}-{end_str}:** 작업자가 안정적인 자세를 유지하며 작업을 수행합니다.")
        
        return "\n".join(summary)
    
    def save_analysis_to_file(self, output_file_path, segment_duration=7):
        """분석 결과를 텍스트 파일로 저장합니다."""
        summary = self.generate_activity_summary(segment_duration)
        
        # 출력 디렉터리 생성
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 파일 저장
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("작업자 활동 분석 결과\n")
            f.write("=" * 50 + "\n\n")
            f.write(summary)
        
        print(f"분석 결과가 '{output_file_path}'에 저장되었습니다.")
        return summary

def main():
    """메인 함수"""
    # JSON 파일 경로 (tf_script.py에서 생성된 파일)
    json_file_path = '../json/output_skeleton_data.json'
    
    # 분석기 생성
    analyzer = SkeletonAnalyzer(json_file_path, fps=30)
    
    # 분석 결과 파일 경로
    output_file_path = '../analysis/worker_activity_analysis.txt'
    
    # 분석 실행 및 저장
    summary = analyzer.save_analysis_to_file(output_file_path, segment_duration=7)
    
    # 콘솔에도 출력
    print("\n" + "=" * 50)
    print("작업자 활동 분석 결과")
    print("=" * 50)
    print(summary)

if __name__ == "__main__":
    main() 