import torch

# Train Encoder Feature Extraction
def encoder_pretrain(model, device, aug_train_loader, optimizer, criterion, epoch):
    model.to(device)
    for batch_idx, (img, aug, _) in enumerate(aug_train_loader):
        # send to device
        img, aug = img.to(device), aug.to(device)

        # forward pass
        optimizer.zero_grad()
        img_output = model(img)
        aug_output = model(aug)

        # calculate difference
        difference_output = torch.sub(img_output, aug_output)
        
        # calculate loss
        loss = criterion(torch.zeros(*difference_output.shape).to(device), difference_output)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch} - Episode: {batch_idx} - Loss: {loss.item()}")