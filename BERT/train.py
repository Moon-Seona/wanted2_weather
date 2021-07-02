# from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score

import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from tqdm import tqdm

import os

import sys

sys.path.append("./")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels_ids ={}
for i in range(46):
    labels_ids[i]=i

def padding(aa,max_len,padding_value=3):

    rows = []
    for a in aa:
        rows.append(np.pad(a, (0, max_len), 'constant', constant_values=padding_value)[:max_len])
    return np.concatenate(rows, axis=0).reshape(-1, max_len)

batch_size = 16 #32 #64 #512
epochs = 10
max_length = 200 # max: 512

class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask.

    It uses a given tokenizer and label encoder to convert any text and labels to numbers that
    can go straight into a GPT2 model.
    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed
    straight into the model - `model(**batch)`.
    Arguments:
      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.
      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to
          labels names and Values map to number associated to those labels.
      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.
    """

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this
        class as a function.
        Arguments:
          item (:obj:`list`):
              List of texts and labels.
        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]

        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,
                                    max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels': torch.tensor(labels)})

        return inputs


class ProblemDataset(Dataset):
    def __init__(self, path, use_tokenizer, val):

        self.texts = []
        self.labels = []
        # Since the labels are defined by folders with data we loop
        # through each label.
        data = pd.read_csv(path).fillna('error')
        end=int(data.__len__()*0.8)
        if val:
            data=data[end:]
        else:
            data=data[:end]
        data['label']=data['label'].astype(int)

        for i in range(len(data)):
            # print(i)
            # print(data.iloc[i]['과제명'],"과제명",data.iloc[i]['요약문_연구내용'],"연구내용")
            self.texts.append(data.iloc[i]['과제명']+data.iloc[i]['요약문_연구내용'])
            # Save encode labels.
            #self.labels.append(str(data.iloc[i]['label']))
            self.labels.append(data.iloc[i]['label'])

        # Number of exmaples.
        self.n_examples = len(self.labels)

        return

    def __len__(self):
        r"""When used `len` return the number of examples.
        """

        return self.n_examples

    def __getitem__(self, item):
        r"""Given an index return an example from the position.

        Arguments:
          item (:obj:`int`):
              Index position to pick an example to return.
        Returns:
          :obj:`Dict[str, str]`: Dictionary of inputs that contain text and
          asociated labels.
        """

        return {'text': self.texts[item],
                'label': self.labels[item]}
class L1Trainer():
    def __init__(self):

        tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
          bos_token='</s>', eos_token='</s>', unk_token='<unk>',
          pad_token='<pad>', mask_token='<mask>')

        path = '~/data/weather2/open/'
        data = path + 'train.csv'
        train_dataset = ProblemDataset(data, True, val=False)
        val_dataset = ProblemDataset(data, True, val=True)

        gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                                  labels_encoder=labels_ids,
                                                                  max_sequence_len=max_length)


        # target = np.array([str(data_item['label']) for data_item in train_dataset])
        # print(target)
        #
        # class_sample_count = np.array(
        #     [len(np.where(target == t)[0]) for t in np.unique(target)])
        #
        # samples_weight = 1. / class_sample_count
        # #samples_weight = np.array([weight[t] for t in target])
        #
        # samples_weight = torch.from_numpy(samples_weight)
        # samples_weight = samples_weight.double()
        # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        # print(class_sample_count)
        # print(samples_weight)



        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
        self.valid_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)

        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='skt/kogpt2-base-v2', num_labels=len(set(labels_ids.values())))

        self.model = GPT2ForSequenceClassification.from_pretrained('skt/kogpt2-base-v2', config=model_config).to(device)

    def train(self,dataloader, optimizer_, scheduler_, device_):

        # Use global variable for model.


        # Tracking variables.
        predictions_labels = []
        true_labels = []
        # Total loss for this epoch.
        total_loss = 0

        # Put the model into training mode.
        self.model.train()

        # For each batch of training data...
        for batch in tqdm(dataloader, total=len(dataloader)):

            # Add original labels - use later for evaluation.
            true_labels += batch['labels'].numpy().flatten().tolist()

            # move batch to device
            batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

            # Always clear any previously calculated gradients before performing a
            # backward pass.
            self.model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this a bert model function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = self.model(**batch)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple along with the logits. We will use logits
            # later to calculate training accuracy.
            loss, logits = outputs[:2]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer_.step()

            # Update the learning rate.
            scheduler_.step()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Convert these logits to list of predicted labels values.
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()

            # Calculate the average loss over the training data.
            avg_epoch_loss = total_loss / len(dataloader)

        # Return all true labels and prediction for future evaluations.
        return true_labels, predictions_labels, avg_epoch_loss



    def validation(self,dataloader, device_):


        # Tracking variables
        predictions_labels = []
        true_labels = []
        #total loss for this epoch.
        total_loss = 0

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.model.eval()

        # Evaluate data for one epoch
        for batch in tqdm(dataloader, total=len(dataloader)):

            # add original labels
            true_labels += batch['labels'].numpy().flatten().tolist()

            # move batch to device
            batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = self.model(**batch)

                # The call to `model` always returns a tuple, so we need to pull the
                # loss value out of the tuple along with the logits. We will use logits
                # later to to calculate training accuracy.
                loss, logits = outputs[:2]

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_loss += loss.item()

                # get predicitons to list
                predict_content = logits.argmax(axis=-1).flatten().tolist()

                # update list
                predictions_labels += predict_content

        # Calculate the average loss over the training data.
        avg_epoch_loss = total_loss / len(dataloader)

        # Return all true labels and prediciton for future evaluations.
        return true_labels, predictions_labels, avg_epoch_loss
    def ProblemTrain(self):

        optimizer = AdamW(self.model.parameters(),
                          lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                          eps = 1e-8 # default is 1e-8.
                          )

        # Total number of training steps is number of batches * number of epochs.
        # `train_dataloader` contains batched data so `len(train_dataloader)` gives
        # us the number of batches.
        total_steps = len(self.train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = len(self.train_dataloader)*2, # Default value in run_glue.py
                                                    num_training_steps = total_steps)
        # Loop through each epoch.
        print('Epoch')
        best_val_acc = 0
        for epoch in tqdm(range(epochs)):
            print()

            print('Training on batches...')
            # Perform one full pass over the training set.
            train_labels, train_predict, train_loss = self.train(self.train_dataloader, optimizer, scheduler, device)
            train_acc = accuracy_score(train_labels, train_predict)

            # Get prediction form model on validation data.
            print('Validation on batches...')
            valid_labels, valid_predict, val_loss = self.validation(self.valid_dataloader, device)
            val_acc = accuracy_score(valid_labels, valid_predict)

            # Print loss and accuracy values to see how training evolves.
            print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
            print()
        #
        # print(valid_labels)
        # print(valid_predict)
        # print(train_labels)
        # print(train_predict)
        self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(),"./save/model.pt")
        print("model saved!")
    def load_model(self):
        self.model.load_state_dict(torch.load("./save/model.pt"))
        print("model loaded!")
    def test_acc(self):
        valid_labels, valid_predict, val_loss = self.validation(self.valid_dataloader, device)
        val_acc = accuracy_score(valid_labels, valid_predict)
        print(" val_loss: %.5f  - valid_acc: %.5f" % (val_loss,  val_acc))

class level1classifier():
    def __init__(self, model_rel_path="./"):
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='classifier/save/kogpt2-base-v2',
                                                  num_labels=len(set(labels_ids.values())))


        self.model_rel_path = model_rel_path

        self.model = GPT2ForSequenceClassification(config=model_config).to(device)

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("classifier/save/kogpt2-base-v2",
                                                            bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                            pad_token='<pad>', mask_token='<mask>')

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_rel_path + "classifier/save/model.pt"))
        print("model loaded!")

def main():
    model=L1Trainer()
    model.ProblemTrain()

if __name__ == "__main__":
    print()
    main()