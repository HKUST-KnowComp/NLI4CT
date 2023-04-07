import torch
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def eval(model, loader, device, print_on_screen=True):
    prediction = []
    ground_label = []
    total_loss = 0
    with torch.no_grad():
        for id, (sent, label) in enumerate(loader):
            label = label.to(device)
            output = model(sent, label, device)
            total_loss += output[0].item()
            pred = torch.argmax(output[1], dim=-1)
            ground_label.append(label)
            prediction.append(pred)
        prediction = torch.stack(prediction).view(-1).cpu().numpy()
        ground_label = torch.stack(ground_label).view(-1).cpu().numpy()
        f_score = f1_score(ground_label, prediction)
        p_score = precision_score(ground_label, prediction)
        r_score = recall_score(ground_label, prediction)
        if print_on_screen:
            print('F1:{:f}'.format(f_score))
            print('precision_score:{:f}'.format(p_score))
            print('recall_score:{:f}'.format(r_score))
    return total_loss, f_score, p_score, r_score
