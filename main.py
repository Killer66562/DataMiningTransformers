from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
from accelerate import Accelerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from optuna.storages import RDBStorage

import numpy as np
import matplotlib.pyplot as plt
import evaluate
import optuna
import wandb


# Static seed
seed=42
set_seed(seed)

# Static train parameters
train_batch_size=16
eval_batch_size=16
gradient_accumulation_steps=4
train_epochs=10

trial_counts=10

project_name="ag-news"
study_name="ag_news_transformers_study"
trial_run_name="ag_news_trial_run"
final_run_name="final_run_with_best_hparams"

model_id = "textattack/distilbert-base-uncased-ag-news"
db_url="sqlite:///optuna_trials.db"

def main():
    # For hyperparameter tuning
    storage = RDBStorage(db_url)
    study = optuna.create_study(
        study_name=study_name, 
        direction="maximize", 
        storage=storage, 
        load_if_exists=True
    )

    # This needs to create an account on wandb.ai and enter an API key manually
    wandb.init(project=project_name, name=study_name)

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.00001, 0.005, log=True), 
            "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.02, log=True),
        }

    def compute_objective(metrics):
        return metrics["eval_accuracy"]
    
    # This needs to create an account on wandb.ai and enter an API key manually
    # wandb.init(project="ag-news", name="ag_news_transformers_study")

    # Load dataset and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset("ag_news")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Preprocess dataset
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    dataset = dataset.map(preprocess_function, batched=True)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Define model init function and load model
    accelerator = Accelerator(mixed_precision="fp16")

    def model_init():
        model_ = AutoModelForSequenceClassification.from_pretrained(model_id)
        return accelerator.prepare_model(model_)

    # Load eval functions and define metrics computation function
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    # Define training args for hyperparameter tuning
    training_args = TrainingArguments(
        output_dir="models", 
        per_device_train_batch_size=train_batch_size, 
        per_device_eval_batch_size=eval_batch_size,  
        gradient_accumulation_steps=gradient_accumulation_steps, 
        num_train_epochs=train_epochs, 
        eval_strategy="epoch", 
        save_strategy="epoch", 
        load_best_model_at_end=True, 
        metric_for_best_model="accuracy", 
        logging_strategy="steps", 
        run_name=trial_run_name, 
        fp16=True, 
        seed=seed
    )

    # Define trainer for hyperparameter tuning
    trainer = Trainer(
        model_init=model_init, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=test_dataset, 
        processing_class=tokenizer, 
        data_collator=data_collator, 
        compute_metrics=compute_metrics
    )
    
    # Get the best run and its hyperparameters
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=trial_counts,
        compute_objective=compute_objective,
        study_name=study_name,
        storage=db_url,
        load_if_exists=True
    )
    best_hparams = best_run.hyperparameters
    print(best_hparams)

    # Define training args for actual training
    training_args = TrainingArguments(
        output_dir="final_model", 
        learning_rate=best_hparams["learning_rate"], 
        per_device_train_batch_size=train_batch_size, 
        per_device_eval_batch_size=eval_batch_size, 
        gradient_accumulation_steps=gradient_accumulation_steps, 
        weight_decay=best_hparams["weight_decay"], 
        eval_strategy="epoch", 
        save_strategy="epoch", 
        load_best_model_at_end=True, 
        num_train_epochs=train_epochs, 
        run_name=final_run_name, 
        logging_strategy="steps", 
        fp16=True, 
        seed=seed
    )

    # Define trainer for actual training
    trainer = Trainer(
        model_init=model_init, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=test_dataset, 
        processing_class=tokenizer, 
        data_collator=data_collator, 
        compute_metrics=compute_metrics
    )

    # Train and save model
    trainer.train()
    trainer.save_model("final_model")
    
    # Test the model and draw confusion matrix
    predictions_output = trainer.predict(test_dataset)
    predicted_logits = predictions_output.predictions
    actual_labels = predictions_output.label_ids
    predicted_labels = np.argmax(predicted_logits, axis=1)
    acc_dict = accuracy.compute(predictions=predicted_labels, references=actual_labels)
    acc = acc_dict['accuracy']
    cm = confusion_matrix(actual_labels, predicted_labels)
    print(f"Accuracy: {acc}")

    label_names = ["World", "Sports", "Business", "Sci/Tech"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)

    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
