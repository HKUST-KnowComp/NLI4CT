SepToken = {'bert': "[SEP]", 'roberta': "</s>", 'electra': "[SEP]", 'deberta': "[SEP]", 'bart': "<s>",
            'gpt2': ''}  # bart we use bos as sep as the code will count the num of eos

Label2num = {'Entailment': 1, "Contradiction": 0}
