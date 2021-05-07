import torch
from tqdm import tqdm


def train_fn(model, data_loader, optimizer, scheduler, epoch, device):
    model.train()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc = "Training epoch: " + str(epoch+1))

    for t,data in enumerate(tk):
        optimizer.zero_grad()
        for k,v in data.items():
            data[k] = v.to(device)

        _, loss = model(**data)
        loss.backward()
        optimizer.step() 
        fin_loss += loss.item() 

        tk.set_postfix({'loss' : '%.6f' %float(fin_loss/(t+1)), 'LR' : optimizer.param_groups[0]['lr']})

    scheduler.step()
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader, epoch, device):
    model.eval()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc = "Validation epoch: " + str(epoch+1))

    with torch.no_grad():
        for t,data in enumerate(tk):
            for k,v in data.items():
                data[k] = v.to(device)

            _, loss = model(**data)
            fin_loss += loss.item() 

            tk.set_postfix({'loss' : '%.6f' %float(fin_loss/(t+1))})
        return fin_loss / len(data_loader)