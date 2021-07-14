# AlexNet
AlexNet implemented with pytorch

![Untitled](https://user-images.githubusercontent.com/55650445/125065687-16d71f00-e0ed-11eb-945a-687a1f521193.png)


### C1
nn.Conv2d(3, 96, k=11, stride=4, padding=0)  
nn.ReLU()  
nn.LocalResponseNorm(5)  
nn.MaxPool2d(k=3, stride=2)  

### C2
nn.Conv2d(96, 256, k=5, stride=1, padding=2)  
nn.ReLU()  
nn.LocalResponseNorm(5)  
nn.MaxPool2d(k=3, stride=2)  

### C3
nn.Conv2d(256, 384, k=3, stride=1, padding=1)  
nn.ReLU()  

### C4
nn.Conv2d(384, 384, k=3, stride=1, padding=1)  
nn.ReLU()  

### C5
nn.Conv2d(384, 256, k=3, stride=1, padding=1)  
nn.ReLU()  
nn.MaxPool2d(k=3, stride=2)  

### F1
nn.Dropout()  
nn.Linear(9216, 4096)  
nn.ReLU()  

### F2
nn.Dropout()  
nn.Linear(4096, 4096)  
nn.ReLU()  

### F3
nn.Linear(4096, num_classes)  
