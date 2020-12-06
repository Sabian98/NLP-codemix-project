import os
import time
import torch
import random
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW, AdapterConfig, AdapterType
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModelWithHeads
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModelWithHeads
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaConfig, XLMRobertaModelWithHeads

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# training
# tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# testing
tokenizer = XLMRobertaTokenizer.from_pretrained('./models/base-xlm-roberta-base-transli-preprocessedv2')

max_len = 56

df = pd.read_csv("data/hinglish_data/SemEval20/Hinglish_dev_3k_split_conll_transliterated_preprocessed.txt", delimiter='\t', header=None, names=['id', 'sentence','label', 'label2'])

print('Number of validating sentences: {:,}\n'.format(df.shape[0]))

sentences = df.sentence.values
labels = df.label.values
input_ids = []
attention_masks = []

for i, sent in enumerate(sentences):
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        add_special_tokens = True,
                        max_length = max_len,
                        padding = 'max_length',
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation = True,
                   )
      
    input_ids.append(encoded_dict['input_ids'])
    
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

val_dataset = TensorDataset(input_ids, attention_masks, labels)

df = pd.read_csv("data/hinglish_data/SemEval20/Hinglish_train_14k_split_conll_transliterated_preprocessed.txt", delimiter='\t', header=None, names=['id', 'sentence','label', 'label2'])

print('Number of training sentences: {:,}\n'.format(df.shape[0]))

sentences = df.sentence.values
labels = df.label.values
input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        add_special_tokens = True,
                        max_length = max_len,
                        padding = 'max_length',
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation = True,
                   )
      
    input_ids.append(encoded_dict['input_ids'])
    
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

train_dataset = TensorDataset(input_ids, attention_masks, labels)

df = pd.read_csv("data/hinglish_data/SemEval20/Hinglish_test_labeled_conll_updated_transliterated_preprocessedv2.txt", delimiter='\t', header=None, names=['id', 'sentence','label', 'label2'])

print('Number of test sentences: {:,}\n'.format(df.shape[0]))

sentences = df.sentence.values
labels = df.label.values
input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        add_special_tokens = True,
                        max_length = max_len,
                        padding = 'max_length',
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation = True,
                   )
      
    input_ids.append(encoded_dict['input_ids'])
    
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

test_dataset = TensorDataset(input_ids, attention_masks, labels)

batch_size = 2

train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size
        )

prediction_dataloader = DataLoader(
            test_dataset,
            sampler = SequentialSampler(test_dataset),
            batch_size = batch_size
        )

# training-adapter
# model = XLMRobertaForSequenceClassification.from_pretrained(
#     "xlm-roberta-base",
#     num_labels = 3, 
#     output_attentions = False,
#     output_hidden_states = False,
# )

# model.add_adapter("base-xlm-roberta-base-transli-preprocessedv2", AdapterType.text_task, config="pfeiffer")
# model.train_adapter(["base-xlm-roberta-base-transli-preprocessedv2"])
# model.set_active_adapters("base-xlm-roberta-base-transli-preprocessedv2")

# testing-adapter
model = XLMRobertaForSequenceClassification.from_pretrained("./models/base-xlm-roberta-base-transli-preprocessedv2")
model.set_active_adapters("base-xlm-roberta-base-transli-preprocessedv2")

model.cuda()

optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )

epochs = 15
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def training():
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []

    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):

            if step % 100 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()        

            loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)            
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        print("")
        print("Running Validation...")

        t0 = time.time()
        model.eval()
        pred_flat = []
        labels_flat = []
        total_eval_loss = 0

        for batch in validation_dataloader:
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad():        
                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                
            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat.extend(np.argmax(logits, axis=1).flatten())
            labels_flat.extend(label_ids.flatten())

        print(metrics.classification_report(pred_flat, labels_flat, digits=3))

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    output_dir = './models/base-xlm-roberta-base-transli-preprocessedv2'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model")

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')

    sns.set(style='darkgrid')

    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    plt.show()


def testing():
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    t0 = time.time()
    model.eval()
    pred_flat = []
    labels_flat = []

    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat.extend(np.argmax(logits, axis=1).flatten())
        labels_flat.extend(label_ids.flatten())

    print(metrics.classification_report(pred_flat, labels_flat, digits=3))

    test_time = format_time(time.time() - t0)
    
    print("  Testing took: {:}".format(test_time))
    print('    DONE.')


# training()

testing()