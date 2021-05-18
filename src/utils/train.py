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


# Train Encoder Feature Extraction
def reform_train(model, device, train_loader, optimizer, criterion, epoch):
    total_loss = 0
    
    model.to(device)

    dist = torch.zeros(10).to(device)
    size = 0
    for batch_idx, (img, _) in enumerate(train_loader):
        # send to device
        img = img.to(device)

        # forward pass
        #optimizer.zero_grad()
        
        img_embedding, _, img_class = model(img)
        classes = torch.argmax(img_class, dim=1).to(device)
        #print(classes.shape)
        size += classes.shape[0]
        #torch.cat((dist, ))
        for c in classes:
            #print(c)
            dist[c] += 1

        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch} - Episode: {batch_idx} - Loss: ")

        loss = torch.nn.functional.kl_div(dist/size, torch.ones(10).to(device)/10)
        loss.requires_grad = True
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss/len(train_loader)

def classifier_train(model, device, train_dataset, optimizer, criterion, epoch, batch_size=20):
    total_loss = 0
    
    model.to(device)
    knn = KNN().to(device)

    for b in range(1):
        batch_dist = torch.zeros(10).to(device)
        for item_idx, (img, _) in enumerate(random.sample(list(train_dataset), batch_size)):
            # send to device
            img = img.to(device).unsqueeze(0)

            # forward pass
            optimizer.zero_grad()
            img_embedding, _, img_class = model(img)
            print(torch.exp(img_class))

            # get samples
            samples = random.sample(list(train_dataset), 500)

          
            for s in range(len(samples)):
                s_img, _ = samples[s]
                embedding, _, s_class = model(s_img.to(device).unsqueeze(0))
                samples[s] = (embedding, s_class)

            # calculate loss
            # intial loss from model forward to one-hot
            desired_one_hot = torch.Tensor([torch.argmax(img_class, dim=1).item()]).to(device)
            loss = 0.5 * torch.nn.functional.nll_loss(img_class, desired_one_hot.long())
            print(loss.item())
            
            # generate stochastic nearest KNN for closest encodings
            neighbors = knn(img_embedding, samples, direction=1)
            n_loss = 0
            for (n_embed, n_class) in neighbors:
                n_loss += 0.3 * torch.nn.functional.nll_loss(n_class, desired_one_hot.long())

            print(n_loss.item())
            
            neighbors = knn(img_embedding, samples, direction=-1)
            f_loss = 0
            
            def random_nn():
                exclude=[torch.argmax(img_class, dim=1).item()]
                randInt = random.randint(0,9)
                return random_nn() if randInt in exclude else randInt 
            
            out_d = torch.Tensor([random_nn()]).to(device)
            for (n_embed, n_class) in neighbors:
                if torch.argmax(n_class, dim=1).item() != torch.argmax(img_class, dim=1).item():
                    out = torch.Tensor([torch.argmax(n_class, dim=1).item()]).to(device)
                    f_loss -= 0.5 * torch.nn.functional.nll_loss(n_class, desired_one_hot.long())
                    f_loss += 0.3 * torch.nn.functional.nll_loss(n_class, out.long())
                    f_loss += 0.5 * torch.nn.functional.nll_loss(n_class, out_d.long())
                else:
                    f_loss -= 0.5 * torch.nn.functional.nll_loss(n_class, desired_one_hot.long())
                    f_loss += 0.5 * torch.nn.functional.nll_loss(n_class, out_d.long())

            print(f_loss.item())

            loss += f_loss + n_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # batch entropy
            batch_dist[torch.argmax(img_class, dim=1).item()]+=1

            #if item_idx % 100 == 0:
            print(f"Epoch: {epoch} - Episode: {(b * 10) + item_idx} - Loss: {loss.item()}")

        optimizer.zero_grad()
        print(batch_dist)
        loss = 100 * batch_size * torch.nn.functional.kl_div(torch.nn.functional.log_softmax(batch_dist/batch_size), torch.ones(batch_dist.shape).to(device)/batch_size)
        print(loss.item())
        loss.requires_grad = True
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss/len(train_dataset)