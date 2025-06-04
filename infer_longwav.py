from kimia_infer.api.kimia import KimiAudio
import os
import soundfile as sf
import argparse
import json
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import tempfile
import time
import warnings
import librosa
from typing import List, Tuple, Dict

warnings.filterwarnings("ignore")

class AudioSegmenter:
    """音频分割器，使用VAD进行智能切分"""
    
    def __init__(self, target_duration=30, min_silence_duration=2.0, sample_rate=16000):
        self.target_duration = target_duration
        self.min_silence_duration = min_silence_duration
        self.sample_rate = sample_rate
        self.vad_model = None
        self.utils = None
        
    def load_vad_model(self):
        """加载Silero VAD模型"""
        print("Loading Silero VAD model...")
        self.vad_model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            verbose=False
        )
        print("VAD model loaded successfully!")
        
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """加载音频文件并重采样"""
        print(f"Loading audio file: {audio_path}")
        
        # 使用librosa加载音频，支持更多格式
        audio, orig_sr = librosa.load(audio_path, sr=None, mono=True)
        
        # 如果需要重采样
        if orig_sr != self.sample_rate:
            print(f"Resampling from {orig_sr}Hz to {self.sample_rate}Hz")
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
        
        # 转换为torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        duration = len(audio_tensor) / self.sample_rate
        print(f"Audio duration: {duration:.2f} seconds")
        
        return audio_tensor, self.sample_rate
    
    def find_optimal_split_points(self, speech_timestamps: List[Dict], 
                                 audio_length: int) -> List[int]:
        """找出最优的切分点"""
        silence_regions = []
        
        # 找出所有静音区域
        if not speech_timestamps:
            return [0, audio_length]
        
        # 开始处的静音
        if speech_timestamps[0]['start'] > 0:
            silence_regions.append({
                'start': 0,
                'end': speech_timestamps[0]['start'],
                'center': speech_timestamps[0]['start'] // 2
            })
        
        # 语音段之间的静音
        for i in range(len(speech_timestamps) - 1):
            silence_start = speech_timestamps[i]['end']
            silence_end = speech_timestamps[i + 1]['start']
            silence_duration = (silence_end - silence_start) / self.sample_rate
            
            if silence_duration >= self.min_silence_duration:
                silence_regions.append({
                    'start': silence_start,
                    'end': silence_end,
                    'center': (silence_start + silence_end) // 2
                })
        
        # 结束处的静音
        if speech_timestamps[-1]['end'] < audio_length:
            silence_regions.append({
                'start': speech_timestamps[-1]['end'],
                'end': audio_length,
                'center': (speech_timestamps[-1]['end'] + audio_length) // 2
            })
        
        # 计算切分点
        split_points = [0]
        current_pos = 0
        target_samples = int(self.target_duration * self.sample_rate)
        
        while current_pos < audio_length - target_samples // 2:
            # 理想的下一个切分点
            ideal_next = current_pos + target_samples
            
            # 在理想位置附近找静音区域
            search_start = ideal_next - int(10 * self.sample_rate)
            search_end = ideal_next + int(10 * self.sample_rate)
            
            # 找到搜索范围内的静音区域
            nearby_silences = [s for s in silence_regions 
                              if search_start <= s['center'] <= search_end 
                              and s['center'] > current_pos]
            
            if nearby_silences:
                # 选择最接近理想位置的静音中心
                best_silence = min(nearby_silences, key=lambda x: abs(x['center'] - ideal_next))
                split_point = best_silence['center']
            else:
                # 如果没有找到合适的静音，强制在理想位置切分
                split_point = ideal_next
            
            split_points.append(split_point)
            current_pos = split_point
        
        # 添加结束点
        if split_points[-1] < audio_length:
            split_points.append(audio_length)
        
        return split_points
    
    def segment_audio(self, audio_path: str) -> List[Dict]:
        """对音频进行分段"""
        if self.vad_model is None:
            self.load_vad_model()
        
        # 加载音频
        audio_tensor, sr = self.load_audio(audio_path)
        audio_length = len(audio_tensor)
        
        print("Detecting speech segments...")
        # 获取语音时间戳
        get_speech_timestamps = self.utils[0]
        speech_timestamps = get_speech_timestamps(
            audio_tensor, 
            self.vad_model, 
            sampling_rate=sr,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100
        )
        
        print(f"Found {len(speech_timestamps)} speech segments")
        
        # 找出最优切分点
        split_points = self.find_optimal_split_points(speech_timestamps, audio_length)
        
        # 创建片段
        segments = []
        temp_dir = tempfile.mkdtemp()
        
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            
            # 提取片段
            segment_audio = audio_tensor[start:end]
            
            # 保存到临时文件
            temp_file = os.path.join(temp_dir, f"segment_{i:04d}.wav")
            torchaudio.save(temp_file, segment_audio.unsqueeze(0), sr)
            
            segment_info = {
                'index': i,
                'audio_path': temp_file,
                'start_time': start / sr,
                'end_time': end / sr,
                'duration': (end - start) / sr,
                'start_sample': start,
                'end_sample': end
            }
            segments.append(segment_info)
            
        print(f"Created {len(segments)} segments")
        return segments, temp_dir

def process_with_kimi(model, segment: Dict, sampling_params: Dict) -> Dict:
    """使用Kimi模型处理单个音频片段"""
    messages = [
        {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
        {
            "role": "user",
            "message_type": "audio",
            "content": segment['audio_path'],
        },
    ]
    
    start_time = time.time()
    try:
        wav, text = model.generate(messages, **sampling_params, output_type="text")
        inference_time = time.time() - start_time
        
        return {
            'success': True,
            'text': text.strip(),
            'inference_time': inference_time
        }
    except Exception as e:
        return {
            'success': False,
            'text': f"[ERROR: {str(e)}]",
            'inference_time': time.time() - start_time,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Kimi Audio inference with VAD-based splitting")
    parser.add_argument("--model_path", type=str, default="moonshotai/Kimi-Audio-7B-Instruct")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file")
    parser.add_argument("--output_path", type=str, default="/opt/data/nvme4/kimi/data/output/output_vad_result.json", help="Path to save the output")
    parser.add_argument("--target_duration", type=float, default=30, help="Target duration for each segment (seconds)")
    parser.add_argument("--min_silence_duration", type=float, default=2.0, help="Minimum silence duration for splitting (seconds)")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate for VAD processing")
    parser.add_argument("--save_segments", action='store_true', help="Save individual segment audio files")
    parser.add_argument("--gpu_device", type=int, default=0, help="GPU device to use")
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device)
    
    # 加载Kimi Audio模型
    print(f"Loading Kimi Audio model from {args.model_path}...")
    model = KimiAudio(
        model_path=args.model_path,
        load_detokenizer=True,
    )
    print("Model loaded successfully!")
    
    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }
    
    # 创建音频分割器
    segmenter = AudioSegmenter(
        target_duration=args.target_duration,
        min_silence_duration=args.min_silence_duration,
        sample_rate=args.sample_rate
    )
    
    # 分割音频
    print(f"\nSegmenting audio file...")
    segments, temp_dir = segmenter.segment_audio(args.audio_path)
    
    # 处理每个片段
    print(f"\nProcessing {len(segments)} segments...")
    results = []
    total_start_time = time.time()
    
    for segment in tqdm(segments, desc="Processing segments"):
        result = process_with_kimi(model, segment, sampling_params)
        
        # 合并结果
        segment_result = {
            'segment_id': segment['index'] + 1,
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'duration': segment['duration'],
            'text': result['text'],
            'inference_time': result['inference_time'],
            'success': result['success']
        }
        
        if not result['success']:
            segment_result['error'] = result.get('error', 'Unknown error')
        
        results.append(segment_result)
    
    total_processing_time = time.time() - total_start_time
    
    # 清理临时文件
    if not args.save_segments:
        print("\nCleaning up temporary files...")
        import shutil
        shutil.rmtree(temp_dir)
    else:
        print(f"\nSegment audio files saved in: {temp_dir}")
    
    # 拼接所有文本
    successful_results = [r for r in results if r['success']]
    full_text = " ".join([r['text'] for r in successful_results])
    
    # 计算统计信息
    total_audio_duration = sum(r['duration'] for r in results)
    total_inference_time = sum(r['inference_time'] for r in results)
    success_rate = len(successful_results) / len(results) * 100 if results else 0
    
    # 准备输出数据
    output_data = {
        'audio_path': args.audio_path,
        'processing_time': total_processing_time,
        'statistics': {
            'total_segments': len(results),
            'successful_segments': len(successful_results),
            'failed_segments': len(results) - len(successful_results),
            'success_rate': success_rate,
            'total_audio_duration': total_audio_duration,
            'total_inference_time': total_inference_time,
            'average_inference_time': total_inference_time / len(results) if results else 0,
            'processing_speed': total_audio_duration / total_processing_time if total_processing_time > 0 else 0
        },
        'full_text': full_text,
        'segments': results,
        'parameters': {
            'model_path': args.model_path,
            'target_duration': args.target_duration,
            'min_silence_duration': args.min_silence_duration,
            'sample_rate': args.sample_rate
        }
    }
    
    # 保存结果
    print(f"\nSaving results to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 打印摘要
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Audio file: {args.audio_path}")
    print(f"Total segments: {len(results)}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total audio duration: {total_audio_duration:.2f} seconds")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    print(f"Processing speed: {output_data['statistics']['processing_speed']:.2f}x realtime")
    print(f"Average inference time per segment: {output_data['statistics']['average_inference_time']:.2f} seconds")
    print(f"Output saved to: {args.output_path}")
    
    # 打印转录结果预览
    print("\n" + "="*50)
    print("TRANSCRIPTION PREVIEW (first 500 characters)")
    print("="*50)
    preview = full_text[:500] + "..." if len(full_text) > 500 else full_text
    print(preview)
    
    # 如果有失败的片段，打印错误信息
    failed_segments = [r for r in results if not r['success']]
    if failed_segments:
        print("\n" + "="*50)
        print("FAILED SEGMENTS")
        print("="*50)
        for seg in failed_segments:
            print(f"Segment {seg['segment_id']}: {seg.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()