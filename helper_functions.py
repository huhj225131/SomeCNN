import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

#Compute model's accuracy
def accuracy_fn(y_test, y_true):
  y_test_labels = torch.argmax(y_test, dim=1)
  correct  = (y_test_labels == y_true).sum().item()
  return round(correct / len(y_true) * 100, 4)

#Train function
def train_model(model, train_dataloader, loss_fn, accuracy_fn, optimizer, device='cpu'):
  model.to(device)
  train_loss, train_acc = 0, 0
  for batch, (X, y) in enumerate(train_dataloader):
    X, y = X.to(device), y.to(device)
    model.train()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_acc += accuracy_fn(torch.softmax(y_pred, dim = 1), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return train_loss / len(train_dataloader), train_acc / len(train_dataloader)

#Evaluation function
def eval_model(model, test_dataloader, loss_fn, accuracy_fn, device='cpu'):
  model.to(device)
  test_loss, test_acc = 0, 0
  for X ,y in test_dataloader:
    X, y = X.to(device), y.to(device)
    model.eval()
    with torch.inference_mode():
      y_pred = model(X)
      test_loss += loss_fn(y_pred, y)
      test_acc += accuracy_fn(torch.softmax(y_pred, dim = 1), y)
  return test_loss/len(test_dataloader), test_acc / len(test_dataloader)

#Create a train-test loop, save the best model that have highest test accuracy during training
def train_test_model(model, train_dataloader,test_dataloader, loss_fn, accuracy_fn, optimizer,path, epochs = 3, device='cpu',best_acc = 0 ):
  torch.save(obj=model.state_dict(),f = path)
  train_acc_per_epoch = []
  test_acc_per_epoch = []
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_model(model, train_dataloader, loss_fn, accuracy_fn, optimizer, device)
    test_loss, test_acc = eval_model(model, test_dataloader, loss_fn, accuracy_fn, device)
    if best_acc < test_acc:
        best_acc = test_acc
        torch.save(obj=model.state_dict(),f = path)
        print("model tốt đấy, lưu nhé")
    train_acc_per_epoch.append(train_acc)
    test_acc_per_epoch.append(test_acc)
    print(f"Epoch: {epoch} | Train loss: {train_loss} | Train accuracy: {train_acc} | Test loss: {test_loss} | Test_acc: {test_acc}")
  return train_acc_per_epoch, test_acc_per_epoch

def plot_trainning_progress(train, test):
  pic = plt.figure(figsize=(10,9))
  pic.add_subplot(1, 2, 1)
  plt.plot(train, label="Train accuracy")
  plt.legend()
  pic.add_subplot(1,2,2)
  plt.plot(test,label = "Test accuracy", c='orange')
  plt.legend()
  plt.show()

# Xavier initialization
def init_cnn(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

# He initialization
# Base on output channels to compute variance (fan_out), and with relu variance = 2 / fan_out
# Usually use for CNN model, with Fully Connected use fan_in
def init_cnn(module):
   if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu') 
    
