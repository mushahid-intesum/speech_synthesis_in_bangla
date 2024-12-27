import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Word:
    id: int          
    text: str        
    head: int        
    deprel: str      
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'text': self.text,
            'head': self.head,
            'deprel': self.deprel,
        }

@dataclass
class Sentence:
    words: List[Word]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'words': [word.to_dict() for word in self.words]
        }

@dataclass
class Document:
    sentences: List[Sentence]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sentences': [sent.to_dict() for sent in self.sentences]
        }

class BengaliDependencyParserModel(nn.Module):
    def __init__(self, bert_model_name='sagorsarker/bangla-bert-base', num_labels=37):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        self.arc_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.arc_d = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.label_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.label_d = nn.Linear(self.hidden_size, self.hidden_size)
        self.label_classifier = nn.Linear(2 * self.hidden_size, num_labels)
        
        # Updated label map with Bengali dependency relations
        self.label_map = {
            0: "root",
            1: "কর্তা",  # subject
            2: "কর্ম",   # object
            3: "নির্ধারক", # determiner
            4: "অনুসর্গ",  # case marker
            5: "বিশেষণ",   # modifier
        }
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        
        arc_h = self.arc_h(sequence_output)
        arc_d = self.arc_d(sequence_output)
        
        arc_scores = torch.einsum('bih,bjh->bij', arc_h, arc_d)
        
        label_h = self.label_h(sequence_output)
        label_d = self.label_d(sequence_output)
        
        label_features = torch.cat([label_h.unsqueeze(2).expand(-1, -1, sequence_output.size(1), -1),
                                  label_d.unsqueeze(1).expand(-1, sequence_output.size(1), -1, -1)], dim=-1)
        
        label_scores = self.label_classifier(label_features)
        
        return arc_scores, label_scores

class BengaliDependencyParser:
    def __init__(self, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained('sagorsarker/bangla-bert-base')
        self.model = BengaliDependencyParserModel()

        self.device = device

    def parse(self, sentence) -> Document:
        """Parse a Bengali sentence and return document structure"""
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Tokenize and prepare input
        encoding = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        try:
            # Get predictions
            with torch.no_grad():
                arc_scores, label_scores = self.model(input_ids, attention_mask)
            
            # Move tensors to CPU and get predictions
            arc_scores = arc_scores.cpu()
            label_scores = label_scores.cpu()
            
            # Get head predictions for each token
            heads = arc_scores[0].argmax(dim=1).tolist()
            
            # Get label predictions
            seq_len = arc_scores.size(1)
            label_scores = label_scores[0]
            
            # Initialize list to store labels
            labels = []
            for i, head in enumerate(heads):
                label_score = label_scores[i][head]
                label = label_score.argmax().item()
                labels.append(label)
            
            # Get original tokens
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            # Convert to Document format
            words = []
            for i, (token, head, label) in enumerate(zip(tokens, heads, labels)):
                # Skip special tokens ([CLS], [SEP], [PAD])
                if token in self.tokenizer.special_tokens_map.values():
                    continue
                    
                word = Word(
                    id=i + 1,
                    text=token,
                    head=head + 1 if head != 0 else 0,
                    deprel=self.model.label_map.get(label, "dep")
                )
                words.append(word)
            
            return Document(sentences=[Sentence(words=words)])
        
        except Exception as e:
            print(f"Error during parsing: {e}")
            raise


def main():
    try:
        # Initialize tokenizer and model
        # tokenizer = AutoTokenizer.from_pretrained('sagorsarker/bangla-bert-base')
        # model = BengaliDependencyParserModel()

        parser = BengaliDependencyParser()
        
        # Example Bengali sentence
        sentence = "আমি বাংলায় কথা বলি।"  # "I speak in Bengali"
        doc = parser.parse(sentence)
        
        # Print results
        print("Document structure:")
        for sent_idx, sentence in enumerate(doc.sentences):
            print(f"\nSentence {sent_idx + 1}:")
            for word in sentence.words:
                print(f"id: {word.id}\ttext: {word.text}\thead: {word.head}\tdeprel: {word.deprel}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()