import torch

def train_one_epoch(model, loader, criterion, optimiser):
    model.train()
    running_loss = 0.0

    for x_b, y_b in loader:
        optimiser.zero_grad()
        preds = model(x_b)
        loss = criterion(preds, y_b)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    
    with torch.inference_mode():
        for x_b, y_b in loader:
            preds = model(x_b)
            loss = criterion(preds, y_b)
            running_loss += loss.item()
    
    return running_loss / len(loader)