import torch
from transformers import AutoModelForSequenceClassification
from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer, GPT2Tokenizer


class Model(torch.nn.Module):
    def __init__(self,args,model_name,from_check_point = False,tokenizer_dir = None, model_dir = None): #if model name is a dir, then we directly load the weight, else we load from transformer package
        super(Model,self).__init__()
        assert(type(from_check_point) == bool)   #Check the datatype

        self.args = args
        if 'gpt' in model_name:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name,do_lower_case = True) if from_check_point == False else GPT2Tokenizer.from_pretrained(tokenizer_dir,do_lower_case = True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels = 2)
            if not from_check_point:
                self.tokenizer.add_special_tokens({'pad_token':'[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            if from_check_point:
                config = torch.load(model_dir,map_location = {'cuda:0':"cuda:0"})
                self.model.load_state_dict(config)

        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels = 2)
            if from_check_point:
                config = torch.load(model_dir)
                self.model.load_state_dict(config)
            #self.model = torch.load(model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,do_lower_case = True) if from_check_point == False else AutoTokenizer.from_pretrained(tokenizer_dir,do_lower_case = True)
        
    def forward(self,sent,label,device):
        token = self.tokenizer(sent, padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device)
        output = self.model(**token,labels = label)

        return output 
    
    def save_model(self,dir):
        self.tokenizer.save_pretrained(dir)
        torch.save(self.model.state_dict(),dir+ f"/dev_best_seed{self.args.seed}.pth")