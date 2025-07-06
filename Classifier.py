import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

"""
实时功率分类器
根据当日及前一天的原始数据，进行处理后，利用已经训练的LeNet5模型分类输出当前所处状态
"""

class DataPreprocessor:
    """数据预处理器 - 基于preprocess.py的DataProcessor实现"""
    
    def __init__(self):
        self.data = None

    def _load_data(self, data:pd.DataFrame):
        """从CSV文件加载数据"""
        try:
            self.data = data.copy()
            self.data['timestamp'] = pd.to_datetime(self.data['TimeStamp'], unit='s')
            self.data['power'] = self.data['P']
        except Exception as e:
            raise ValueError(f"加载数据失败: {e}")

    def get_day_data(self, date_str: str, resample: bool = True, denoise: bool = True, interval: int = 300):
        """获取指定日期的一天内所有功率数据"""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            beijing_day_start = date.timestamp()
            beijing_day_end = beijing_day_start + 24 * 3600
            
            day_data = self.data[
                (self.data['timestamp'].apply(lambda x: x.timestamp()) >= beijing_day_start) & 
                (self.data['timestamp'].apply(lambda x: x.timestamp()) < beijing_day_end)
            ]
            
            if day_data.empty:
                raise ValueError(f"指定日期 {date_str} 无数据")

            start_ts = date.timestamp()
            time_array = (day_data['timestamp'].apply(lambda x: x.timestamp()) - start_ts).tolist()
            power_array = day_data['power'].tolist()

            if denoise:
                power_array = self.denoise_data(power_array)

            if resample:
                return self.resample_data(time_array, power_array, interval)
            return time_array, power_array

        except Exception as e:
            raise ValueError(f"获取当天数据失败: {e}")

    def denoise_data(self, power_array: list, method: str = 'mean', window_size: int = 5) -> list:
        """对功率数据进行去噪处理"""
        try:
            series = pd.Series(power_array)
            if method == 'mean':
                return series.rolling(window=window_size, min_periods=1).mean().tolist()
            elif method == 'median':
                return series.rolling(window=window_size, min_periods=1).median().tolist()
            else:
                raise ValueError("不支持的去噪方法，仅支持 'mean' 或 'median'")
        except Exception as e:
            raise ValueError(f"去噪失败: {e}")

    def resample_data(self, time_array: list, power_array: list, interval: int = 300):
        """将非等间隔时间序列重采样为等间隔"""
        try:
            ts = pd.Series(power_array, index=pd.to_timedelta(time_array, unit='s'))
            res = ts.resample(f'{interval}s').mean().interpolate()
            return [x.total_seconds() for x in res.index], res.tolist()
        except Exception as e:
            raise ValueError(f"线性重采样失败: {e}")

    def get_realtime_data(self, current_time: datetime, sequence_length: int = 96, interval: int = 300):
        """获取实时数据用于分类"""
        try:
            # 计算需要的历史数据时间范围
            history_minutes = (sequence_length - 1) * 5  # 序列长度对应的分钟数
            start_time = current_time - timedelta(minutes=history_minutes)
            
            # 获取时间范围内的数据
            window_data = self.data[
                (self.data['timestamp'] >= start_time) & 
                (self.data['timestamp'] <= current_time)
            ].sort_values('timestamp')
            
            if len(window_data) < sequence_length:
                return None
            
            # 重采样到5分钟间隔
            start_ts = start_time.timestamp()
            time_array = (window_data['timestamp'].apply(lambda x: x.timestamp()) - start_ts).tolist()
            power_array = window_data['power'].tolist()
            
            # 去噪和重采样
            power_array = self.denoise_data(power_array)
            resampled_time, resampled_power = self.resample_data(time_array, power_array, interval)
            
            # 确保序列长度一致
            if len(resampled_power) >= sequence_length:
                return np.array(resampled_power[-sequence_length:])
            else:
                return None
                
        except Exception as e:
            print(f"获取实时数据失败: {e}")
            return None


class LeNet5(nn.Module):
    """LeNet5模型实现"""
    
    def __init__(self, num_classes=10, input_size=64):
        super(LeNet5, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.sig = nn.Sigmoid()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        if input_size == 64:
            fc_input_size = 16 * 14 * 14
        elif input_size == 28:
            fc_input_size = 16 * 5 * 5
        else:
            size_after_conv1 = input_size
            size_after_pool1 = size_after_conv1 // 2
            size_after_conv2 = size_after_pool1 - 4
            size_after_pool2 = size_after_conv2 // 2
            fc_input_size = 16 * size_after_pool2 * size_after_pool2
        
        self.f5 = nn.Linear(fc_input_size, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sig(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.sig(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.sig(x)
        x = self.f6(x)
        x = self.sig(x)
        x = self.f7(x)
        return x


def GAF_Transform(data: np.ndarray, method: str = 'summation') -> np.ndarray:
    """GAF变换实现"""
    length = data.shape[0]
    scaled_data = 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1
    theta_data = np.arccos(scaled_data)
    matrix = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            if method == 'summation':
                matrix[i][j] = np.cos(theta_data[i] + theta_data[j])
            else:
                matrix[i][j] = np.sin(theta_data[i] - theta_data[j])
    return matrix


class PowerClassifier:
    """功率分类器主类"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 model_path: str = 'final_power_classification_model.pth',
                 sequence_length: int = 96,
                 image_size: int = 64):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.image_size = image_size
        
        # 标签映射
        self.label_mapping = {
            'mode_1': '无人活动',
            'mode_2': '轻度办公', 
            'mode_3': '餐饮休息',
            'mode_4': '标准办公',
            'mode_5': '夜间办公',
            'mode_6': '深夜空闲',
            'mode_7': '周末活跃',
            'mode_8': '周末空闲'
        }
        
        # 类别标签列表（按照训练时的顺序）
        self.class_labels = ['mode_1', 'mode_2', 'mode_3', 'mode_4', 'mode_5', 'mode_6', 'mode_7', 'mode_8']
        
        self.preprocessor = None
        self.model = None
        
        print(f">> 使用设备: {self.device}")
        print(f">> 序列长度: {self.sequence_length}, 图像大小: {self.image_size}")
        
        # 初始化组件
        self._init_preprocessor(data)
        self._init_model()

    def _init_preprocessor(self, data: pd.DataFrame):
        """初始化数据预处理器"""
        try:
            self.preprocessor = DataPreprocessor()
            self.preprocessor._load_data(data)
            print("[SUCCESS] 数据预处理器初始化完成")
        except Exception as e:
            print(f"[ERROR] 数据预处理器初始化失败: {e}")
            raise

    def _init_model(self):
        """初始化模型"""
        try:
            # 创建模型
            num_classes = len(self.class_labels)
            self.model = LeNet5(num_classes=num_classes, input_size=self.image_size).to(self.device)
            
            # 加载权重
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print("[SUCCESS] 成功加载模型权重")
                else:
                    self.model.load_state_dict(checkpoint)
                    print("[SUCCESS] 成功加载模型权重")
                
                self.model.eval()
                print("[SUCCESS] LeNet5模型初始化完成")
            else:
                print(f"[ERROR] 模型文件不存在: {self.model_path}")
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
                
        except Exception as e:
            print(f"[ERROR] 模型初始化失败: {e}")
            raise

    def predict_single_sequence(self, power_sequence: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        分类单个功率序列
        Args:
            power_sequence (np.ndarray): 输入的功率序列，形状为 (sequence_length,)
        Returns:
            Tuple[str, float, np.ndarray]: 预测的标签、置信度和各类别概率
        """
        try:
            if self.model is None:
                print("[ERROR] 模型未初始化")
                return 'mode_1', 0.0, np.zeros(len(self.class_labels))
                
            with torch.no_grad():
                # 确保序列长度一致
                if len(power_sequence) > self.sequence_length:
                    power_sequence = power_sequence[-self.sequence_length:]
                elif len(power_sequence) < self.sequence_length:
                    padding = np.full(self.sequence_length - len(power_sequence), power_sequence[-1])
                    power_sequence = np.concatenate([power_sequence, padding])
                
                # 转换为GAF图像
                gaf_matrix = GAF_Transform(power_sequence, method='summation')
                
                # 调整图像大小
                if gaf_matrix.shape[0] != self.image_size:
                    gaf_matrix = cv2.resize(gaf_matrix, (self.image_size, self.image_size))
                
                # 归一化到[0, 1]
                gaf_matrix = (gaf_matrix - gaf_matrix.min()) / (gaf_matrix.max() - gaf_matrix.min())
                
                # 转换为张量
                image = torch.FloatTensor(gaf_matrix).unsqueeze(0).unsqueeze(0).to(self.device)
                
                # 模型预测
                output = self.model(image)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                predicted_idx = np.argmax(probabilities)
                predicted_label = self.class_labels[predicted_idx]
                confidence = probabilities[predicted_idx]
                
                return predicted_label, confidence, probabilities
                
        except Exception as e:
            print(f"[ERROR] 预测失败: {e}")
            return 'mode_1', 0.0, np.zeros(len(self.class_labels))

    def classify_daily_data(self, target_date: str) -> List[Dict]:
        """分类当日零点到实时的数据
        需要使用当日及前一天的数据来构建完整的序列进行预测
        Args:
            target_date (str): 目标日期，格式为 'YYYY-MM-DD'
        Returns:
            List[Dict]: 包含每个时刻的分类结果
        """
        print(f"\n>> 开始分类日期: {target_date}")
        
        try:
            # 计算前一天的日期
            target_datetime = datetime.strptime(target_date, '%Y-%m-%d')
            previous_date = (target_datetime - timedelta(days=1)).strftime('%Y-%m-%d')
            
            print(f">> 获取前一天数据: {previous_date}")
            
            # 获取前一天的数据（需要末尾部分用于构建序列）
            try:
                prev_time_array, prev_power_array = self.preprocessor.get_day_data(
                    previous_date, resample=True, denoise=True, interval=300
                )
                print(f">> 前一天数据点数: {len(prev_power_array)}")
            except Exception as e:
                print(f"[WARNING] 无法获取前一天数据: {e}，将只使用当日数据")
                prev_power_array = []
            
            # 获取当日数据（5分钟间隔重采样）
            print(f">> 获取当日数据: {target_date}")
            time_array, power_array = self.preprocessor.get_day_data(
                target_date, resample=True, denoise=True, interval=300
            )
            
            if not time_array or not power_array:
                print(f"[ERROR] 无法获取日期 {target_date} 的数据")
                return []
            
            print(f">> 当日数据点数: {len(power_array)}")
            
            # 合并前一天末尾和当日数据
            # 取前一天最后 (sequence_length-1) 个点，确保能构建完整序列
            if prev_power_array:
                # 取前一天末尾的数据点
                prev_tail_length = min(len(prev_power_array), self.sequence_length - 1)
                prev_tail = prev_power_array[-prev_tail_length:]
                
                # 合并数据：前一天末尾 + 当日全部
                combined_power = prev_tail + power_array
                print(f">> 合并后总数据点数: {len(combined_power)} (前一天末尾: {len(prev_tail)}, 当日: {len(power_array)})")
                
                # 调整时间索引，当日数据从合并数组中的对应位置开始
                current_day_start_idx = len(prev_tail)
            else:
                # 如果没有前一天数据，只使用当日数据
                combined_power = power_array
                current_day_start_idx = 0
                print(f">> 仅使用当日数据: {len(combined_power)} 个点")
            
            # 计算当前时刻（距离最近的5分钟整点）
            current_time = datetime.now()
            current_minutes = current_time.hour * 60 + current_time.minute
            # 向下取整到最近的5分钟
            rounded_minutes = (current_minutes // 5) * 5
            current_5min_point = rounded_minutes // 5  # 转换为5分钟点的索引
            
            # 只处理到当前时刻的当日数据点
            valid_points = min(len(power_array), current_5min_point + 1)
            
            results = []
            
            for i in range(valid_points):
                # 当日第i个点在合并数组中的索引
                combined_idx = current_day_start_idx + i
                
                # 确保有足够的历史数据进行预测
                if combined_idx >= self.sequence_length - 1:
                    # 获取用于预测的序列（前96个点）
                    start_idx = combined_idx - self.sequence_length + 1
                    sequence = np.array(combined_power[start_idx:combined_idx + 1])
                    
                    # 进行预测
                    predicted_label, confidence, probabilities = self.predict_single_sequence(sequence)
                else:
                    # 数据不足时，使用默认模式
                    predicted_label = 'mode_1'  # 默认为无人活动
                    confidence = 0.5
                    probabilities = np.ones(len(self.class_labels)) / len(self.class_labels)
                    print(f"   [WARNING] 时间点 {i} 历史数据不足，使用默认模式")
                
                # 计算时间
                time_in_seconds = time_array[i]
                hours = int(time_in_seconds // 3600)
                minutes = int((time_in_seconds % 3600) // 60)
                time_str = f"{hours:02d}:{minutes:02d}"
                
                result = {
                    'time_point': i,
                    'time_str': time_str,
                    'time_seconds': time_in_seconds,
                    'power': power_array[i],
                    'predicted_mode': predicted_label,
                    'mode_description': self.label_mapping.get(predicted_label, predicted_label),
                    'confidence': confidence,
                    'probabilities': probabilities.tolist(),
                    'has_sufficient_history': combined_idx >= self.sequence_length - 1
                }
                
                results.append(result)
                
                print(f"   {time_str}: {self.label_mapping.get(predicted_label, predicted_label)} (置信度: {confidence:.3f})")
            
            print(f"\n[SUCCESS] 完成分类，共 {len(results)} 个时刻")
            return results
            
        except Exception as e:
            print(f"[ERROR] 分类过程失败: {e}")
            return []

    def get_current_status(self) -> Dict:
        """获取当前实时状态"""
        try:
            current_time = datetime.now()
            
            # 获取实时数据序列
            power_sequence = self.preprocessor.get_realtime_data(current_time, self.sequence_length)
            
            if power_sequence is None:
                print("[WARNING] 无法获取足够的实时数据")
                return {
                    'status': 'insufficient_data',
                    'message': '数据不足，无法进行分类'
                }
            
            # 进行预测
            predicted_label, confidence, probabilities = self.predict_single_sequence(power_sequence)
            
            result = {
                'status': 'success',
                'current_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_mode': predicted_label,
                'mode_description': self.label_mapping.get(predicted_label, predicted_label),
                'confidence': confidence,
                'probabilities': {label: prob for label, prob in zip(self.class_labels, probabilities.tolist())}
            }
            
            return result
            
        except Exception as e:
            print(f"[ERROR] 获取当前状态失败: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def print_classification_summary(self, results: List[Dict]):
        """打印分类结果摘要"""
        if not results:
            print("无分类结果")
            return
        
        print(f"\n{'='*60}")
        print(f"分类结果摘要 (共 {len(results)} 个时刻)")
        print(f"{'='*60}")
        
        # 统计各模式的出现次数
        mode_counts = {}
        for result in results:
            mode = result['predicted_mode']
            mode_desc = result['mode_description']
            key = f"{mode} ({mode_desc})"
            mode_counts[key] = mode_counts.get(key, 0) + 1
        
        print("\n模式分布:")
        for mode, count in sorted(mode_counts.items()):
            percentage = count / len(results) * 100
            print(f"  {mode}: {count} 次 ({percentage:.1f}%)")
        
        print(f"\n最新状态:")
        latest = results[-1]
        print(f"  时间: {latest['time_str']}")
        print(f"  模式: {latest['mode_description']}")
        print(f"  置信度: {latest['confidence']:.3f}")


def main():
    """主函数 - 演示分类器使用"""
    data = pd.read_csv('data.csv')


    # 初始化分类器
    classifier = PowerClassifier(
        model_path='final_power_classification_model.pth',
        data = data,
    )
    
    # 获取今天的日期
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\n使用日期: {today}")
    today = '2025-06-28'
    
    # 分类当日数据
    results = classifier.classify_daily_data(today)
    
    # 打印结果摘要
    classifier.print_classification_summary(results)

    
    print(results)
    
    # 获取当前实时状态
    print(f"\n{'='*60}")
    print("当前实时状态")
    print(f"{'='*60}")
    current_status = classifier.get_current_status()
    
    if current_status['status'] == 'success':
        print(f"当前时间: {current_status['current_time']}")
        print(f"预测模式: {current_status['mode_description']}")
        print(f"置信度: {current_status['confidence']:.3f}")
        print("\n各模式概率:")
        for mode, prob in current_status['probabilities'].items():
            mode_desc = classifier.label_mapping.get(mode, mode)
            print(f"  {mode_desc}: {prob:.3f}")
    else:
        print(f"状态: {current_status['status']}")
        print(f"消息: {current_status.get('message', '未知错误')}")


if __name__ == "__main__":
    main()