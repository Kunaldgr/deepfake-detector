# import torch
# from torchvision import transforms
# from PIL import Image
# import numpy as np

# # Same preprocessing as training
# def get_transform():
#     return transforms.Compose([
#         transforms.Resize((112, 112)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])

# def preprocess_image(image):
#     """
#     Preprocess PIL image for model input
    
#     Args:
#         image: PIL Image
    
#     Returns:
#         Preprocessed tensor
#     """
#     transform = get_transform()
    
#     # Convert to RGB if needed
#     if image.mode != 'RGB':
#         image = image.convert('RGB')
    
#     # Apply transforms
#     img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
#     return img_tensor

# def predict(model, image_tensor, device):
#     """
#     Run prediction on image
    
#     Args:
#         model: Trained model
#         image_tensor: Preprocessed image tensor
#         device: CPU or CUDA
    
#     Returns:
#         Dictionary with prediction results
#     """
#     model.eval()
    
#     with torch.no_grad():
#         image_tensor = image_tensor.to(device)
#         outputs = model(image_tensor)
        
#         # Apply softmax
#         probs = torch.nn.functional.softmax(outputs, dim=1)
        
#         # Get prediction
#         prediction = torch.argmax(probs, dim=1).item()
#         confidence = probs[0, prediction].item() * 100
        
#         # Get both probabilities
#         # CORRECTED: During training folders were real/fake but dataset assigned them as:
#         # index 0 = FAKE (because dataset creation logic assigned fake first)
#         # index 1 = REAL (because real was assigned second)
#         fake_prob = probs[0, 0].item() * 100
#         real_prob = probs[0, 1].item() * 100
    
#     # DEBUG: Print what model actually outputs
#     print(f"🔍 DEBUG INFO:")
#     print(f"   Raw outputs: {outputs}")
#     print(f"   Probabilities: Fake={fake_prob:.2f}%, Real={real_prob:.2f}%")
#     print(f"   Prediction index: {prediction}")
#     print(f"   Confidence: {confidence:.2f}%")
    
#     # CORRECTED: index 0 = fake, index 1 = real (based on actual model behavior)
#     return {
#         'prediction': 'Real' if prediction == 1 else 'Fake',
#         'confidence': f"{confidence:.2f}%",
#         'probabilities': {
#             'fake': round(fake_prob, 2),
#             'real': round(real_prob, 2)
#         },
#         'is_real': prediction == 1
#     }


import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Same preprocessing as training
def get_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def preprocess_image(image):
    """
    Preprocess PIL image for model input
    
    Args:
        image: PIL Image
    
    Returns:
        Preprocessed tensor
    """
    transform = get_transform()
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return img_tensor

def predict(model, image_tensor, device):
    """
    Run prediction on image
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        device: CPU or CUDA
    
    Returns:
        Dictionary with prediction results
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        
        # Apply softmax
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get prediction
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0, prediction].item() * 100
        
        # Get both probabilities
        prob_idx0 = probs[0, 0].item() * 100
        prob_idx1 = probs[0, 1].item() * 100
    
    # ENHANCED DEBUG
    print(f"\n{'='*60}")
    print(f"🔍 DEBUG INFO:")
    print(f"{'='*60}")
    print(f"   Raw logits: {outputs[0].cpu().numpy()}")
    print(f"   After softmax:")
    print(f"      Index 0: {prob_idx0:.2f}%")
    print(f"      Index 1: {prob_idx1:.2f}%")
    print(f"   Argmax prediction: {prediction}")
    print(f"   Confidence: {confidence:.2f}%")
    
    # FLIPPED MAPPING FIX:
    # The model learned: Index 0 = Real, Index 1 = Fake (opposite!)
    actual_prediction = 'Fake' if prediction == 1 else 'Real'
    print(f"   Final prediction: {actual_prediction}")
    print(f"{'='*60}\n")
    
    return {
        'prediction': actual_prediction,  # FLIPPED!
        'confidence': f"{confidence:.2f}%",
        'probabilities': {
            'real': round(prob_idx0, 2),    # FLIPPED!
            'fake': round(prob_idx1, 2)     # FLIPPED!
        },
        'is_real': prediction == 0  # FLIPPED!
    }