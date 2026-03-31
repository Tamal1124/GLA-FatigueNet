"""Real-time webcam demo for GLA-FatigueNet."""
import os, sys, torch, cv2, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_config, load_checkpoint
from models.gla_fatiguenet import GLAFatigueNet
from data.landmark_extractor import LandmarkExtractor
from data.augmentation import get_inference_transforms


def run_realtime(config_path="config/config.yaml", checkpoint_path=None):
    config = load_config(config_path)
    device = torch.device('cpu')
    model = GLAFatigueNet(config).to(device)
    cp = checkpoint_path or config['inference']['checkpoint_path']
    if os.path.exists(cp):
        load_checkpoint(model, cp)
    model.eval()
    transform = get_inference_transforms(config)
    extractor = LandmarkExtractor(static_image_mode=False, num_features=config['model']['gla']['geometric_features'])
    fatigue_classes = config['data']['fatigue_classes']
    emotion_classes = config['data']['emotion_classes']
    colors = {'alert': (0,255,0), 'drowsy': (0,165,255), 'fatigued': (0,0,255)}
    cap = cv2.VideoCapture(config['inference'].get('webcam_id', 0))
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam"); return
    print("[INFO] Press 'q' to quit")
    fps_counter, fps_start = 0, time.time()
    display_fps = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        geo = extractor.extract_features(rgb)
        transformed = transform(image=rgb)
        img_t = transformed['image'].unsqueeze(0).to(device)
        geo_t = torch.from_numpy(geo).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out = model(img_t, geo_t)
        fat_pred = torch.softmax(out['fatigue_logits'], 1)[0]
        emo_pred = torch.softmax(out['emotion_logits'], 1)[0]
        fat_cls = fatigue_classes[fat_pred.argmax().item()]
        emo_cls = emotion_classes[emo_pred.argmax().item()]
        fat_conf = fat_pred.max().item()
        emo_conf = emo_pred.max().item()
        color = colors.get(fat_cls, (255,255,255))
        cv2.putText(frame, f"Fatigue: {fat_cls} ({fat_conf:.0%})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Emotion: {emo_cls} ({emo_conf:.0%})", (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"EAR: {geo[2]:.3f} MAR: {geo[3]:.3f}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        fps_counter += 1
        if time.time() - fps_start >= 1:
            display_fps = fps_counter; fps_counter = 0; fps_start = time.time()
        cv2.putText(frame, f"FPS: {display_fps}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        cv2.imshow('GLA-FatigueNet Real-Time Demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows(); extractor.close()

if __name__ == '__main__':
    run_realtime()
