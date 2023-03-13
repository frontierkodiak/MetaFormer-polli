from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image
from config import get_inference_config
from models import build_model
from torch.autograd import Variable
from torchvision.transforms import transforms
import numpy as np
import argparse
import os # Import os module

try:
    from apex import amp
except ImportError:
    amp = None

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def model_config(config_path):
    args = Namespace(cfg=config_path)
    config = get_inference_config(args)
    return config


def read_class_names(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    class_list = []

    for l in lines:
        line = l.strip().split()
        # class_list.append(line[0])
        class_list.append(line[1][4:])

    classes = tuple(class_list)
    return classes


class GenerateEmbedding:
    def __init__(self, text_file):
        self.text_file = text_file

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def generate(self):
        text_list = []
        with open(self.text_file, 'r') as f_text:
            for line in f_text:
                line = line.encode(encoding='UTF-8', errors='strict')
                line = line.replace(b'\xef\xbf\xbd\xef\xbf\xbd', b' ')
                line = line.decode('UTF-8', 'strict')
                text_list.append(line)
            # data = f_text.read()
        select_index = np.random.randint(len(text_list))
        inputs = self.tokenizer(text_list[select_index], return_tensors="pt", padding="max_length",
                                truncation=True, max_length=32)
        outputs = self.model(**inputs)
        embedding_1d_tensor = outputs.pooler_output.squeeze(0)

        return embedding_1d_tensor


def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', default='configs/efficientnet_b3.yaml', type=str,
                        help='Path to configuration file.')
    parser.add_argument('--image_dir', default='images/', type=str,
                        help='Path to image directory.') # Add argument for image directory
    args_1d_embedding_generator = parser.parse_args()

    config_1d_embedding_generator_model_path = model_config(args_1d_embedding_generator.config)

    model_1d_embedding_generator_model_path = build_model(config_1d_embedding_generator_model_path)

    checkpoint_1d_embedding_generator_model_path = torch.load(
        config_1d_embedding_generator_model_path.MODEL.CHECKPOINT_PATH,
        map_location=torch.device('cpu'))

    if amp is not None:
        model_1d_embedding_generator_model_path.load_state_dict(checkpoint_1d_embedding_generator_model_path['state_dict'])
        #optimizer.load_state_dict(checkpoint_1d_embedding_generator_model_path['optimizer'])
        amp.load_state_dict(checkpoint_1d_embedding_generator_model_path['amp'])
    else:
        model_1d_embedding_generator_model_path.load_state_dict(checkpoint_1d_embedding_generator_model_path['state_dict'])

    model_1d_embedding_generator_model_path.eval()
    

    # Loop through the image directory recursively using os.walk()
    for root, dirs, files in os.walk(args_1d_embedding_generator.image_dir):
        for filename in files: # Loop through the files in each subdirectory
            if filename.endswith('.jpg') or filename.endswith('.png'): # Check if the file is an image
                image_path = os.path.join(root, filename) # Get the full path of the image
                print(f'Processing {image_path}') # Print a message to indicate progress
                img = Image.open(image_path) # Open the image using PIL
                img = img.convert('RGB') # Convert to RGB mode
                img = img.resize((config_1d_embedding_generator_model_path.MODEL.IMAGE_SIZE,
                                    config_1d_embedding_generator_model_path.MODEL.IMAGE_SIZE)) # Resize to the model input size
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ]) # Define a transformation to normalize the image
                img = transform(img) # Apply the transformation
                img = Variable(img.unsqueeze(0)) # Add a batch dimension and wrap in a Variable

                with torch.no_grad(): # Disable gradient computation
                    output_1d_embedding_generator_model_path = model_1d_embedding_generator_model_path(img) # Perform inference on the image

                output_1d_embedding_generator_model_path = output_1d_embedding_generator_model_path.squeeze(0) # Remove the batch dimension

                embedding_2d_tensor = GenerateEmbedding('text.txt').generate() # Generate a 2D embedding from text

                embedding_concatenated_tensor = torch.cat((output_1d_embedding_generator_model_path,
                                                            embedding_2d_tensor), 0) # Concatenate the 1D and 2D embeddings

                print(embedding_concatenated_tensor.shape) # Print the shape of the concatenated embedding