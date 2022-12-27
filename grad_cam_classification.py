class Extractor():

    def __init__(self, model, target_layer = 'conv1'):
        self.model = model
        self.target_layer = target_layer #layer4.1.conv2
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient=grad

    def __call__(self, x):
        self.gradients = []
        for name,module in self.model._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
            print(name, x.shape)
            if name == self.target_layer:
                x.register_hook(self.save_gradient)
                target_activation=x

        x = self.model.avgpool(x)
        x=x.view(1,-1)
        x = self.model.fc2(x)
        return target_activation, x



class GradCam():
    def __init__(self, model, target_layer_name, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = Extractor(self.model, target_layer_name)

    
    def __call__(self, input, index = None, size = (600,1000)):
        if self.cuda:
            target_activation, output = self.extractor(input.cuda())
        else:
            target_activation, output = self.extractor(input)

        # index is the index of class you are interested in
        # the default is the predicted class
        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        # input single observation
        # so the batch is 1
        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1.0
        one_hot = torch.tensor(one_hot)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        grads_val = self.extractor.gradient.cpu().data.numpy()
         # shape（c, h, w）
        target = target_activation.cpu().data.numpy()[0]
        # shape（c,）
        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        # values less than 0, replaced by 0
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, size)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
