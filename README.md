# Spam Detection using Deep Learning Models

This project focuses on building various deep learning models to detect spam messages using Natural Language Processing techniques. The models implemented in this project include:

1. **Dense Model**: A simple feedforward neural network model.
2. **Long Short Term Memory (LSTM) Model**: A recurrent neural network (RNN) model designed to capture sequential information in text data.
3. **Bi-directional Long Short Term Memory (BiLSTM) Model**: An extension of the LSTM model that learns patterns from both before and after a given token within a document, improving accuracy.
4. **BERT Model**: A state-of-the-art pre-trained model for natural language understanding.

## File Structure

- `spam.ipynb`: Jupyter Notebook containing the Python code for data preprocessing, model building, training, and evaluation.

## Libraries Used

- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `wordcloud`
- `scikit-learn`
- `tensorflow`
- `keras`
- `plotly`
- `tensorflow_hub`
- `tensorflow_text`

## Usage

1. Clone the repository to your local machine.
2. Open `spam.ipynb` in Jupyter Notebook or any compatible environment.
3. Follow the instructions and execute the code cells sequentially to preprocess data, build models, train, and evaluate them.

## Dataset

The dataset used in this project is the `SMSSpamCollection`, containing labeled SMS messages as spam or ham (non-spam).

## Models Implemented

1. **Dense Model**
   - A simple neural network architecture with an embedding layer, global average pooling, dense layers, and dropout for regularization.

2. **LSTM Model**
   - A recurrent neural network architecture using Long Short Term Memory cells to capture sequential patterns in text data.

3. **BiLSTM Model**
   - An extension of the LSTM model that utilizes bidirectional LSTM cells to capture patterns from both directions in text data.

4. **BERT Model**
   - Utilizes a pre-trained BERT model for text encoding and classification, achieving state-of-the-art performance in natural language understanding tasks.


## Conclusion

This project demonstrates the effectiveness of deep learning models in spam detection tasks using SMS messages. The models implemented show comparable performance, with the BERT model being the most advanced, leveraging pre-trained embeddings for superior understanding of text data.
