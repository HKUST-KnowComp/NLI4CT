import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from dataloader import Recorddataset
from evaluate import eval
from model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--ptlm", default='microsoft/deberta-v3-large', type=str)
    parser.add_argument("--lmn", default="deberta", type=str, help="name of the language model")
    parser.add_argument('--data', type=str, help='data dir')
    parser.add_argument("--epoch", default=40, type=int)
    parser.add_argument("--eval_every", default=10, type=int)
    parser.add_argument("--prompt", default=2, choices=[0, 1, 2, 3], type=int)
    parser.add_argument("--mode", default='trn', choices=['mix', 'trn'])
    parser.add_argument("--from_check_point", default=False, type=bool)
    parser.add_argument("--tokenizer_dir", default=None, type=str, help='the tokenizer check point dir')
    parser.add_argument("--model_dir", default=None, type=str, help='the model check point dir')
    parser.add_argument("--seed", default=621, type=int)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    if args.mode == 'trn':
        trn_dataset = Recorddataset(args, args.data, "train")
    else:
        trn_dataset = Recorddataset(args, args.data, "trn&dev")
    dev_dataset = Recorddataset(args, args.data, "dev")
    tst_dataset = Recorddataset(args, args.data, "test")

    trn_loader = DataLoader(trn_dataset, batch_size=4, shuffle=True, drop_last=False)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, drop_last=False)
    tst_loader = DataLoader(tst_dataset, batch_size=4, shuffle=False, drop_last=False)

    seed_val = args.seed
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    output_dir = "./result/{}_prompt{}_mode{}_epoch{}_eval{}/".format(args.ptlm, args.prompt, args.mode, args.epoch,
                                                                          args.eval_every)
    os.makedirs(output_dir, exist_ok=True)
    epochs = args.epoch
    num_total_steps = len(trn_loader) * epochs
    num_warmup_steps = len(trn_loader) * int(args.epoch / 8)

    model = Model(args, args.ptlm, args.from_check_point, args.tokenizer_dir, args.model_dir)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-6, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_total_steps)

    best_val, best_val_epoch = 0, 0
    best_recall, best_precision = 0, 0
    for epoch in range(epochs):
        total_loss = 0
        for iter, (sent, label) in enumerate(
                tqdm(trn_loader, desc=f'epoch: {epoch + 1}/{epochs}')):  # data = (statement, trail1,trail2,label)
            label = label.to(device)
            output = model(sent, label, device)
            pred = torch.argmax(output[1], dim=-1)
            total_loss += output[0].item()
            optimizer.zero_grad()
            output[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if iter % args.eval_every == 0 and iter != 0:
                with torch.no_grad():
                    l, f, p, r = eval(model, dev_loader, device, print_on_screen=False)
                print(
                    f"The Validation result at epoch {epoch + 1} iter {iter}: val_loss: {l}, val_f1: {f}, val_precision: {p}, val_recall: {r}")
                if f > best_val:
                    best_val_epoch = epoch + 1
                    best_val = f
                    best_precision = p
                    best_recall = r
                    model.save_model(output_dir)
                    Test_Results = {}
                    for sent, uuid in tqdm(tst_loader):
                        with torch.no_grad():
                            outputs = model(sent, label, device)
                            output = outputs[1]
                            for i in range(4):
                                if torch.argmax(output[i]) == 0:
                                    Test_Results[str(uuid[i])] = {"Prediction": 'Contradiction'}
                                else:
                                    Test_Results[str(uuid[i])] = {"Prediction": "Entailment"}

                    with open("{}/results.json".format(output_dir), 'w') as jsonFile:
                        jsonFile.write(json.dumps(Test_Results, indent=4))

        print("total_loss_per_epoch: ", total_loss, "best_val", best_val, 'best_r', best_recall, 'best_p',
              best_precision, "best_val_epoch", best_val_epoch)
        if args.mode == 'mix':
            Test_Results = {}
            for (sent, uuid) in tqdm(tst_loader):
                outputs = model(sent, label, device)
                output = outputs[1]
                for i in range(4):
                    if torch.argmax(output[i]) == 0:
                        Test_Results[str(uuid[i])] = {"Prediction": 'Contradiction'}
                    else:
                        Test_Results[str(uuid[i])] = {"Prediction": "Entailment"}

            with open("{}/epoch{}_results.json".format(output_dir, epoch + 1), 'w') as jsonFile:
                jsonFile.write(json.dumps(Test_Results, indent=4))


if __name__ == '__main__':
    main()
