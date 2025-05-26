# Fine-Tuning-DistilBERT-with-LoRA-for-Text-Classification
This project demonstrates fine-tuning a pre-trained DistilBERT (67M) model using LoRA (Low-Rank Adaptation) for sentiment analysis on a truncated IMDb dataset. It leverages Hugging Face's transformers, peft, and datasets libraries to build an efficient text classification pipeline.


# Fine-Tuning DistilBERT with LoRA for Text Classification  

## 🔍 Overview  
This project fine-tunes a DistilBERT model for binary sentiment analysis (positive/negative) using:  
- **LoRA**: Parameter-efficient fine-tuning via the `peft` library.  
- **Truncated IMDb Dataset**: 2k training/validation samples for rapid experimentation.  
- **Hugging Face Ecosystem**: Transformers, datasets, and evaluation tools.  

## Key Features  
- 🧠 **LoRA Fine-Tuning**: Reduces trainable parameters for faster training.  
- 📦 **Modular Code**: Preprocessing, training, and evaluation pipelines.  
- 🚀 **Ready for Deployment**: Model can be pushed to Hugging Face Hub.  
- 📊 **Evaluation Metrics**: Accuracy and Loss tracking during training.  

## 🛠️ Setup  
### Dependencies  
Install required packages:  
```bash  
pip install transformers datasets peft evaluate torch numpy
````

## 📊 Dataset
* **Dataset Link**: [https://huggingface.co/datasets/shawhin/imdb-truncated]

<p align="center">
  <img src="resume_screening.png" alt="THE APP" width="500"/>
</p>

The dataset (shawhin/imdb-truncated) can be loaded via:
```python
dataset = load_dataset("shawhin/imdb-truncated")  
````

## Usage
Train the Model 
```bash
python fine_tune_lora.ipynb 
````

## ✨ Evaluate Predictions
Compares untrained vs. trained model predictions on sample text.

<p align="center">
  <img src="Screenshot 2025-05-26 124344.png" alt="THE APP" width="500"/>
  <img src="Screenshot 2025-05-26 124235.png" alt="THE APP" width="500"/>
</p>

## 🛠️ Push to Hugging Face Hub (Optional) 
```python
model.push_to_hub("")  
````

## 📄 Results

<p align="center">
  <img src="resume_screening.png" alt="img" width="500"/>
</p>

Accuracy : Trained model achieves ~85% accuracy on the validation set (varies with seed/data).
Efficiency : LoRA reduces trainable parameters by ~97% compared to full fine-tuning.

## Credits
This work is based on [Fine-Tuning Large Language Models (LLMs)](https://medium.com/towards-data-science/fine-tuning-large-language-models-llms-23473d763b91 ).




