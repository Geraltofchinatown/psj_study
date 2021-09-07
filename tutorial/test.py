import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda
from collections.abc import Iterable
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: nn.zeros(64, 10, dtype=nn.float).scatter(0, nn.tenser(y), value=1))
)

testing_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: nn.zeros(64, 10, dtype=nn.float).scatter(0, nn.tenser(y), value=1))
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        preds = nn.softmax(dim=1)(logits)
        y_preds = preds.argmax(1)
        return y_preds

'''
class custom_image(dataset):
    def __init__(self,annotation_file, img_dir, transform = None, target_transform = None):
        self.image_label = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_label)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_label.iloc[idx, 0])
        image = read_file(img_path)
        label = self.image_label.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

'''
training_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
testing_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)
#train_features, train_labels = next(iter(training_dataloader))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NeuralNetwork().to(device)


#loss = nn.functional.binary_cross_entropy_with_logits()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

