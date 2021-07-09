import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import AlexNet



### parameters
model_name = "AlexNet"
path = "D:/projects"
datapath = path + '/dataset'
modelpath = path + "/" + model_name + "/models/" + model_name + "_best_model.h"
batch_size = 32

### 사용 가능한 gpu 확인 및 설정
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)



### test set
# from train_set
meanR, meanG, meanB = 0.4467106, 0.43980986, 0.40664646
stdR, stdG, stdB = 0.22414584, 0.22148906,  0.22389975
# define the image transforamtion for test_set
test_transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB]),
                transforms.Resize(227)
])
# load STL10 test dataset
test_set = datasets.STL10(datapath, split='test', download=True, transform=test_transformer)
print(test_set.data.shape)
# test_set loader
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
# check dataloader
for x,y in test_loader:
    print(x.shape)
    print(y.shape)
    break



### model
model = AlexNet()
model.to(device)



### test function
def test():
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    avg_loss = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            # 순전파
            prediction = model(x)
            correct = prediction.max(1)[1] == y
            test_acc = correct.float().mean()
    print("Acc: [{:.2f}%]".format(
        test_acc*100
    ))


### test
test()