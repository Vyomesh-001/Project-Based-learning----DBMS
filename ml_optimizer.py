import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])

MODEL_NAME = "t5-small"
MODEL_PATH = "./model/t5-query-optimizer"
tokenizer = None
model = None

def train_model():
    """
    Trains and saves the T5 model for query optimization.
    """
    global tokenizer
    df = pd.read_csv('data/query_dataset.csv')
    
    prefix = "optimize sql: "
    df['inefficient_query'] = prefix + df['inefficient_query']

    train_df, val_df = train_test_split(df, test_size=0.1)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    
    train_encodings = tokenizer(list(train_df['inefficient_query']), truncation=True, padding=True)
    val_encodings = tokenizer(list(val_df['inefficient_query']), truncation=True, padding=True)
    
    train_labels = tokenizer(list(train_df['efficient_query']), truncation=True, padding=True)
    val_labels = tokenizer(list(val_df['efficient_query']), truncation=True, padding=True)

    train_dataset = QueryDataset(train_encodings, train_labels)
    val_dataset = QueryDataset(val_encodings, val_labels)
    
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,  # Reduced epochs for faster training
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print("Starting model training...")
    trainer.train()
    print("Training complete!")

    # Save the model and tokenizer
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def optimize_with_ml(sql_query):
    """
    Uses the trained T5 model to optimize a single SQL query.
    """
    global tokenizer, model
    
    if tokenizer is None or model is None:
        try:
            print("Loading ML model...")
            tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
            model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
            print("Model loaded successfully.")
        except OSError:
            print("Error: Model not found. Please train the model first.")
            return None
        except ImportError as e:
            print(f"Error: Missing dependencies for ML optimization: {e}")
            print("ML optimization is not available. Using rule-based optimization only.")
            return None

    prefix = "optimize sql: "
    input_text = prefix + sql_query
    
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        inputs, 
        max_length=150, 
        num_beams=4, 
        early_stopping=True
    )
    
    optimized_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "original_query": sql_query,
        "optimized_query": optimized_query,
        "explanation": "Optimized using a trained T5 transformer model."
    }

if __name__ == '__main__':
    
    train_model()