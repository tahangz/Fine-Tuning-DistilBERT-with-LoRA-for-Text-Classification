# Fine-Tuning-DistilBERT-with-LoRA-for-Text-Classification
This project demonstrates fine-tuning a pre-trained DistilBERT model using LoRA (Low-Rank Adaptation) for sentiment analysis on a truncated IMDb dataset. It leverages Hugging Face's transformers, peft, and datasets libraries to build an efficient text classification pipeline.


# Fine-Tuning DistilBERT with LoRA for Text Classification  
**Blog Reference**: [Fine-Tuning Large Language Models (LLMs)](https://medium.com/towards-data-science/fine-tuning-large-language-models-llms-23473d763b91 )  

## Overview  
This project fine-tunes a DistilBERT model for binary sentiment analysis (positive/negative) using:  
- **LoRA**: Parameter-efficient fine-tuning via the `peft` library.  
- **Truncated IMDb Dataset**: 1k training/validation samples for rapid experimentation.  
- **Hugging Face Ecosystem**: Transformers, datasets, and evaluation tools.  

## Key Features  
- 🧠 **LoRA Fine-Tuning**: Reduces trainable parameters for faster training.  
- 📦 **Modular Code**: Preprocessing, training, and evaluation pipelines.  
- 🚀 **Ready for Deployment**: Model can be pushed to Hugging Face Hub.  
- 📊 **Evaluation Metrics**: Accuracy tracking during training.  

## Setup  
### Dependencies  
Install required packages:  
```bash  
pip install transformers datasets peft evaluate torch numpy
````

## Dataset
* **Dataset Link**: [[https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset]](https://huggingface.co/datasets/shawhin/imdb-truncated)
  
The dataset (shawhin/imdb-truncated) can be loaded via:
```python
dataset = load_dataset("shawhin/imdb-truncated")  
````

## Usage
Train the Model 
python fine_tune_lora.py 


## Evaluate Predictions
Compares untrained vs. trained model predictions on sample text.

## Push to Hugging Face Hub (Optional) 
model.push_to_hub("your-username/distilbert-lora-sentiment")  


## Results
Accuracy : Trained model achieves ~85% accuracy on the validation set (varies with seed/data).
Efficiency : LoRA reduces trainable parameters by ~90% compared to full fine-tuning.

## Credits
This work is based on the Towards Data Science article series on LLMs.




