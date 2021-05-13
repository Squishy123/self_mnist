import torch
from torch.autograd.grad_mode import no_grad
import random

from models.knn import KNN

# Train Encoder Feature Extraction
def encoder_pretrain(model, device, aug_train_loader, optimizer, criterion, epoch):
    total_loss = 0
    
    model.to(device)
    for batch_idx, (img, aug, _) in enumerate(aug_train_loader):
        # send to device
        img, aug = img.to(device), aug.to(device)

        # forward pass
        optimizer.zero_grad()
        img_embedding, img_class = model(img)
        aug_embedding, aug_class = model(aug)

        # calculate difference
        difference_output = torch.sub(img_embedding, aug_embedding)
        
        # calculate loss
        loss = criterion(torch.zeros(*difference_output.shape).to(device), difference_output)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch} - Episode: {batch_idx} - Loss: {loss.item()}")
    
    return total_loss/len(aug_train_loader)

def classifier_train(model, device, train_dataset, optimizer, criterion, epoch):
    total_loss = 0
    
    model.to(device)
    knn = KNN().to(device)
    for item_idx, (img, _) in enumerate(random.sample(list(train_dataset), 50)):
        # send to device
        img = img.to(device).unsqueeze(0)

        # forward pass
        optimizer.zero_grad()
        img_embedding, img_class = model(img)

        # get samples
        samples = random.sample(list(train_dataset), 100)

        with torch.no_grad():
            for s in range(len(samples)):
                s_img, _ = samples[s]
                embedding, s_class = model(s_img.to(device).unsqueeze(0))
                samples[s] = (embedding, s_class)

        # generate stochastic KNN for encodings
        neighbors = knn(img_embedding, samples)

        # calculate loss
        loss = 0
        for (n_embed, n_class) in neighbors:
            #loss += torch.nn.functional.hinge_embedding_loss(n_embed, img_embedding)
            loss += torch.nn.functional.binary_cross_entropy(img_class, n_class.detach())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        #if item_idx % 100 == 0:
        print(f"Epoch: {epoch} - Episode: {item_idx} - Loss: {loss.item()}")
    
    return total_loss/len(train_dataset)