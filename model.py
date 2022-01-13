import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False) # This ffectively freezes the model, by not calculating grad for backprop
        
            modules = list(resnet.children())[:-1] # copying the whole model, except the last layer
        self.resnet = nn.Sequential(*modules) # wrapper that bundles all layers to look like one
        self.embed = nn.Linear(resnet.fc.in_features, embed_size) # FC layer that uses variable output size

    def forward(self, images):
        # I'm used to features being named x...same thing of course
        features = self.resnet(images) # Apply the imported resnet model 
        features = features.view(features.size(0), -1) #  reshaping/flattening/unrolling feature tensor
        features = self.embed(features) # apply the final layer (the custom embedding layer)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.rich_print = False
        if self.rich_print:
            print("Entering DecoderRNN.__init__")
            print("embed_size: ", embed_size)
            print("hidden_size: ", hidden_size)
            print("vocab_size: ", vocab_size)
            print("num_layers: ", num_layers)
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # pytorch terms: (num_embeddings, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size) 
        
        # pytorch terms: (input_sz, hidden_sz, num_layers)
        self.lstm = nn.LSTM(embed_size, 
                            hidden_size, 
                            num_layers, 
                            dropout=0, 
                            batch_first=True) 
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        """
        Inputs: 
            features: The context vector from the Encoder. In this case, the embedded CNN feature vectors.
                Shape: (batch_size, embed_size)
                Type: Tensor
            captions: The idexes for the vocab words in the sample sentences
                Shape: (batch_size, sentence_length) 
                Type: Tensor
        Return: Model predicted outputs
            outputs: shape (batch_size, captions.shape[1], vocab_size)
        
        """
#         if self.rich_print:
#             print("Entering DecoderRnn.forward")
#             batch_size, sentence_length = captions.shape
#             print("sentence_length: ", captions.shape[1])
#             print("batch_size: ", batch_size)
#             print("x input features: ", features.shape) # (batch_size, embed_size)
#             print("x input captions: ", captions.shape) # (batch_size, sentence_length)
        
        # Discard the last word (it's <end>, so making a prediction about the word after that is non-sensical)
        # Embed the remaining captions
        word_embeddings = self.embedding(captions[:, :-1]) # (batch_size, sentence_length, embedding?) #####
        
        # Merge features in front of captions inputs, making the CNN the first input
        # unsqueeze the features, creating an extra dim (sentence_length) along which to concat 
        all_inputs = torch.cat((features.unsqueeze(dim=1), word_embeddings), dim=1)
        
        output, hidden = self.lstm(all_inputs) # bundling tuple(h, c) as features
        
        output = self.fc(output) 
        
#         output = self.softmax(output) 
        
        return output


    def sample(self, inputs, states=None, max_len=20):
        """
        Purpose: This method generates a sample sentence based on "inputs", an image embedding.
        
        Inputs:
            inputs: A Tensor of shape (1,1,embed_size)
            states: a tuple of (hidden, cell) states (the None default will populate 0's initially)
            max_len: Max sentence length before stopping word generation artificially, even if <end> has not been reached.        
        
            
        Outputs:
            sentence: a list of word indices
        
        Instructor-provided description:
            accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)
        
        """
        print("Input shape: ", inputs.shape)
        assert(inputs.shape == (1,1,self.embed_size))
        assert(states==None)
        
        
        # format inputs
        caption_words = []

        for i in range(max_len):
            output, states = self.lstm(inputs, states)
            output_vocab_scores = self.fc(output)
            # output = self.softmax(output) # we could include this, but max(x) = max(softmax(x))
            _,word_tensor = output_vocab_scores.max(2)
            word_idx = word_tensor.item() # pulls the value out of the tensor wrapper
            caption_words.append(word_idx)
            if(word_idx == 1):
                break;
                
            inputs = self.embedding(word_tensor) # prep for next cycle
            
        return caption_words