import torch
from torch.autograd.grad_mode import no_grad
import random

from models.knn import KNN
import numpy as np

# Train Encoder Feature Extraction
def encoder_pretrain(model, device, aug_train_loader, optimizer, criterion, epoch):
    total_loss = 0
    
    model.to(device)
    for batch_idx, (img, aug, _) in enumerate(aug_train_loader):
        # send to device
        img, aug = img.to(device), aug.to(device)

        # forward pass
        optimizer.zero_grad()
        img_embedding, img_decoded, img_class = model(img)
        aug_embedding, _, aug_class = model(aug)

        # calculate difference
        #difference_output = torch.sub(img_embedding, aug_embedding)
        
        # calculate loss
        #loss = torch.nn.functional.mse_loss(difference_output, torch.zeros(*difference_output.shape).to(device))
        loss = torch.nn.functional.mse_loss(img_decoded, img)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch} - Episode: {batch_idx} - Loss: {loss.item()}")
    
    return total_loss/len(aug_train_loader)

def classifier_train(model, device, train_dataset, optimizer, criterion, epoch, batch_size=10):
    total_loss = 0
    
    model.to(device)
    knn = KNN().to(device)

    for b in range(10):
        batch_dist = torch.zeros(10).to(device)
        for item_idx, (img, _) in enumerate(random.sample(list(train_dataset), batch_size)):
            # send to device
            img = img.to(device).unsqueeze(0)

            # forward pass
            optimizer.zero_grad()
            img_embedding, _, img_class = model(img)

            # get samples
            samples = random.sample(list(train_dataset), 100)

            with torch.no_grad():
                for s in range(len(samples)):
                    s_img, _ = samples[s]
                    embedding, _, s_class = model(s_img.to(device).unsqueeze(0))
                    samples[s] = (embedding, s_class)

            # calculate loss
            # intial loss from model forward to one-hot
            #desired_one_hot = torch.zeros(img_class.shape[1], dtype=torch.long).to(device)
            #desired_one_hot[torch.argmax(img_class, dim=1).item()] = 1
            desired_one_hot= torch.tensor([torch.argmax(img_class, dim=1).item()]).to(device)
            loss = torch.nn.functional.cross_entropy(img_class, desired_one_hot)
            
            # generate stochastic nearest KNN for closest encodings
            neighbors = knn(img_embedding, samples)
            for (n_embed, n_class) in neighbors:
                #difference_output = torch.sub(img_embedding, n_embed)
                #loss += torch.nn.functional.mse_loss(difference_output, torch.zeros(*difference_output.shape).to(device))
                #loss += torch.nn.functional.binary_cross_entropy_with_logits(img_class, n_class.detach())
                loss += torch.nn.functional.cross_entropy(n_class, desired_one_hot)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # batch entropy
            batch_dist[torch.argmax(img_class, dim=1).item()]+=1

            #if item_idx % 100 == 0:
            print(f"Epoch: {epoch} - Episode: {(b * 10) + item_idx} - Loss: {loss.item()}")

        optimizer.zero_grad()
        loss = torch.nn.functional.kl_div(batch_dist/batch_size, torch.ones(batch_dist.shape).to(device)/batch_size, reduction="batchmean")
        loss.requires_grad = True
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss/len(train_dataset)