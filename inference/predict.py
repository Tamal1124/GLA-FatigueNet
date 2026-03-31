"""Single image prediction."""
import os, sys, torch, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_config, load_checkpoint
from models.gla_fatiguenet import GLAFatigueNet
from data.landmark_extractor import LandmarkExtractor
from data.augmentation import get_inference_transforms


def predict(image_path, config_path="config/config.yaml", checkpoint_path=None):
    config = load_config(config_path)
    device = torch.device('cpu')
    model = GLAFatigueNet(config).to(device)
    cp = checkpoint_path or config['inference']['checkpoint_path']
    if os.path.exists(cp):
        load_checkpoint(model, cp)
    model.eval()
    transform = get_inference_transforms(config)
    extractor = LandmarkExtractor(num_features=config['model']['gla']['geometric_features'])
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    geo_features = extractor.extract_features(image_rgb)
    transformed = transform(image=image_rgb)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    geo_tensor = torch.from_numpy(geo_features).unsqueeze(0).float().to(device)
    with torch.no_grad():
        outputs = model(img_tensor, geo_tensor)
    fatigue_probs = torch.softmax(outputs['fatigue_logits'], dim=1)[0]
    emotion_probs = torch.softmax(outputs['emotion_logits'], dim=1)[0]
    fatigue_pred = fatigue_probs.argmax().item()
    emotion_pred = emotion_probs.argmax().item()
    fatigue_classes = config['data']['fatigue_classes']
    emotion_classes = config['data']['emotion_classes']
    print(f"\nPrediction for: {image_path}")
    print(f"  Fatigue: {fatigue_classes[fatigue_pred]} ({fatigue_probs[fatigue_pred]:.2%})")
    print(f"  Emotion: {emotion_classes[emotion_pred]} ({emotion_probs[emotion_pred]:.2%})")
    print(f"  Gate Values: {outputs['gate_values']}")
    return {'fatigue': fatigue_classes[fatigue_pred], 'emotion': emotion_classes[emotion_pred],
            'fatigue_probs': fatigue_probs.numpy(), 'emotion_probs': emotion_probs.numpy()}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    predict(args.image, args.config, args.checkpoint)
