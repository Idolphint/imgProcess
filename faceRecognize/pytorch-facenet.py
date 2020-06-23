from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os

data_dir = './FaceRecognitionData/faces94/person/'

batch_size = 32
epochs = 13
workers = 0 if os.name == 'nt' else 8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

#################################先得到场景中的人脸框，其实不需要前面的步骤哦
trans = transforms.Compose([
    transforms.Resize((192,192)),
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = datasets.ImageFolder(data_dir, transform=trans)
name_idx_dict = dataset.class_to_idx
#print(dataset.class_to_idx)
resnet = InceptionResnetV1(
    classify=True,
    num_classes=len(dataset.class_to_idx)
).to(device)




optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]
test_inds = img_inds[int(0.95 * len(img_inds)):]
train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)
test_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=1,
    sampler=SubsetRandomSampler(test_inds)
)


loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}


writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)
resnet.eval()
training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )
###############      show    test      ############

resnet_pt = resnet.eval()
    #embs = resnet_pt(val_loader)
def get_key(dict_, value):
    for k, v in dict_.items():
        if v == value:
            return k
    #return [k for k, v in dict_.items() if v == value]

for i, (img, idx) in enumerate(test_loader, 0):
    idx = idx.numpy()
    acl = get_key(name_idx_dict, idx[0])
    img = img.to(device)
    logits = resnet_pt(img)
    
    logits = logits.cpu().detach().numpy()[0]
    logits = np.argmax(logits)
    res = get_key(name_idx_dict, logits)
    #res = dataset.idx_to_class['logits']
    sample_name, _ = val_loader.dataset.samples[i]
    print("accuary: ", acl, "res: ", res, "name: ", sample_name)

writer.close()


