from collections import Counter
from gensim.models import Word2Vec
from matplotlib.image import imread
from matplotlib.pyplot import figure, imshow, ion, plot, savefig, show, title, xlabel, ylabel
from os import path
from pandas import read_csv
from PIL import Image
from spacy import load
from torch import cat, device, no_grad, tensor
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss, Dropout, Embedding, Linear, LSTM, Module, Sequential
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.models import resnet18

# Dataset and variable loadings
spacyEng, Device, image_dir, captions_dir, SOS, PAD, UNK, EOS, Epochs, Divider, listLosses = load("en_core_web_sm"), device("cuda" if is_available() else "cpu"), "Images", "Captions.txt", "<SOS>", "<PAD>", "<UNK>", "<EOS>", range(20), 2000, []
df = read_csv(captions_dir)
df.head()

imageIndex = 5
imshow(imread(image_dir + "/" + df.iloc[imageIndex, 0]))
show()
for Caption in range(imageIndex, imageIndex + 5):
    print(f"Caption - {df.iloc[Caption, 1]}")
    
class Vocabulary:
    def __init__(self, Threshold):
        self.itos, self.threshold = {0: PAD, 1: SOS, 2: EOS, 3: UNK}, Threshold
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenize(Text):
        return [Token.text.lower() for Token in spacyEng.tokenizer(Text)]

    def build_vocab(self, Sentences):
        Frequencies, Index = Counter(), 4
        for Sentence in Sentences:
            for Word in self.tokenize(Sentence):
                Frequencies[Word] += 1
                
                if Frequencies[Word] == self.threshold:
                    self.stoi[Word], self.itos[Index] = Index, Word
                    Index += 1

    def numericalize(self, Text):
        tokenizedText = self.tokenize(Text)
        return [self.stoi[Token] if Token in self.stoi else self.stoi[UNK] for Token in tokenizedText]

# Alternative way to construct the vocabulary using Word2Vec model
def WordVec(Sentences):
    validResponses = ["CBOW", "SG"]
    while True:
        modelMethod = input("CBOW or Skip-gram? <CBOW, Sg> ").upper()
        if modelMethod in validResponses:
            print("Alright!")
            break
        print("Select one, dude!")

    if modelMethod == "CBOW":
        # Create CBOW model
        Model = Word2Vec(min_count = 5, vector_size = 300, window = 5)
    else:
        # Create Skip-gram model (sg = 1)
        Model = Word2Vec(min_count = 5, sg = 1, vector_size = 300, window = 5)

    Model.build_vocab(Sentences, progress_per = 1000)

    # train on own data
    Model.train(Sentences, epochs = 100, total_examples = len(Sentences))

    return Model.wv

# Testing "Vocabulary" class
Voc = Vocabulary(Threshold = 1)
Voc.build_vocab(["Why this semester doesn't end?", "Oops!", "C'mon dude I'm over..."])
print(f"\n{Voc.stoi}")
print(Voc.numericalize("When does this semester end I'm exhausted dude!"))

class datasetCustomizer(Dataset):
    def __init__(self, root_dir, captions_file, Transform = None, Threshold = 5):
        self.df, self.root_dir = read_csv(captions_file), root_dir

        self.imgs, self.captions, self.transform = self.df["image"], self.df["caption"], Transform
        
        self.vocab = Vocabulary(Threshold)
        self.vocab.build_vocab(self.captions.tolist())
        
        # Word2Vec approach
        #self.vocab = WordVec(self.captions.tolist())
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, Index):
        image, Caption = Image.open(path.join(self.root_dir, self.imgs[Index])).convert("RGB"), self.captions[Index]

        if self.transform != None:
            image = self.transform(image)
        return image, tensor([self.vocab.stoi[SOS]] + self.vocab.numericalize(Caption) + [self.vocab.stoi[EOS]])

class CapsCollate:
    def __init__(self, pad_idx, batchFirst = False):
        self.batch_first, self.pad_idx = batchFirst, pad_idx
        
    def __call__(self, Batch):
        return cat([Item[0].unsqueeze(0) for Item in Batch], dim = 0), pad_sequence([Item[1] for Item in Batch], batch_first = self.batch_first, padding_value = self.pad_idx)

class Encoder(Module):
    def __init__(self, embeddingSize):
        super(Encoder, self).__init__()
        ResNet18 = resnet18(weights = True)
        #ResNet18.load_state_dict(load("ResNet18 Pretrained Weights.pth"))
        ResNet18.eval()

        # Freezing all layers and weights except the fully connected layer.
        for Parameter in ResNet18.parameters():
            Parameter.requires_grad = False

        for Parameter in ResNet18.fc.parameters():
            Parameter.requires_grad = True

        # Unreezing all layers and weights.
        """
        for Parameter in ResNet18.parameters():
            Parameter.requires_grad = True
        """
        summary(ResNet18, input_size = (3, 64, 64), batch_size = -1)
        self.resnet = Sequential(*list(ResNet18.children())[:-1])
        self.embedding = Linear(ResNet18.fc.in_features, embeddingSize)

    def forward(self, Images):
        Features = self.resnet(Images)
        return self.embedding(Features.view(Features.size(0), -1))

class Decoder(Module):
    def __init__(self, embeddingSize, hiddenSize, vocabularySize, numberLayers = 1, dropProbability = 0.3):
        super(Decoder, self).__init__()
        self.embedding, self.lstm, self.fcn, self.drop = Embedding(vocabularySize, embeddingSize), LSTM(embeddingSize, hiddenSize, num_layers = numberLayers, batch_first = True), Linear(hiddenSize, vocabularySize), Dropout(dropProbability)

    def forward(self, Features, captions):
        x, _ = self.lstm(cat((Features.unsqueeze(1), self.embedding(captions[:,:-1])), dim = 1))
        return self.fcn(x)

    # Testing time!
    def captionGenerator(self, Inputs, Hidden = None, maximumLength = 20, Vocab = None):
        # Inference part
        # Given the image features generate the captions
        captions = []
        
        for i in range(maximumLength):
            Output, Hidden = self.lstm(Inputs, Hidden)
            Output = self.fcn(Output).view(Inputs.size(0), -1)
            
            #select the word with most val
            predictedIndex = Output.argmax(dim = 1)
            
            #save the generated word
            captions.append(predictedIndex.item())
            
            #end if <EOS detected>
            if Vocab.itos[predictedIndex.item()] == EOS:
                break
            
            #send generated word as the next caption
            Inputs = self.embedding(predictedIndex.unsqueeze(0))
        
        #covert the vocab idx to words and return sentence
        return [Vocab.itos[Index] for Index in captions]

class encoderDecoder(Module):
    def __init__(self, embeddingSize, hiddenSize, vocabularySize, numberLayers = 1, dropProbability = 0.3):
        super(encoderDecoder, self).__init__()
        self.encoder, self.decoder = Encoder(embeddingSize), Decoder(embeddingSize, hiddenSize, vocabularySize, numberLayers, dropProbability)

    def forward(self, images, captions): 
        return self.decoder(self.encoder(images), captions)

# Defining the transform to be applied
def Illustrator(Input, Title = None):
    imshow(Input.numpy().transpose((1, 2, 0)))
    if Title != None:
        title(Title)
    show()

# Dataset construction
Dataset = datasetCustomizer(root_dir = image_dir, captions_file = captions_dir, Transform = Compose([Resize((224, 224)), ToTensor()]))
vocab = Dataset.vocab
PI, vocabularySize = vocab.stoi[PAD], len(vocab)
Dataset = DataLoader(dataset = Dataset, batch_size = 4, num_workers = 1, shuffle = True, collate_fn = CapsCollate(pad_idx = PI, batchFirst = True), pin_memory = True)

# Initializing model, loss and optimizer
Model, loss, datasetSize = encoderDecoder(300, 256, vocabularySize, 2).to(Device), CrossEntropyLoss(ignore_index = PI), len(Dataset)
Optimizer = Adam(Model.parameters(), lr = 0.0001)

# main()
for Epoch in Epochs:
    Losses = 0
    for Index, (image, captions) in enumerate(iter(Dataset)):
        image, captions = image.to(Device), captions.to(Device)
        
        # Zero the gradients.
        Optimizer.zero_grad()

        # Feed forward
        Outputs = Model(image, captions)

        # Calculate the batch loss.
        Loss = loss(Outputs.view(-1, vocabularySize), captions.view(-1))

        # Backward pass.
        Loss.backward()

        # Update the parameters in the optimizer.
        Optimizer.step()
        
        Loss = Loss.item()
        Losses += Loss

        if (Index + 1) % Divider == 0:
            print("In Epoch {} with loss {:.5f}".format(Epoch + 1, Loss))
            Model.eval()
            with no_grad():
                dataIter = iter(Dataset)
                image, _ = next(dataIter)
                Features = Model.encoder(image[0: 1].to(Device))
                Illustrator(image[0], Title = " ".join(Model.decoder.captionGenerator(Features.unsqueeze(0), Vocab = vocab)))
           
            # (FINALLY) training part!
            Model.train()

    listLosses.append(Losses / datasetSize)

print(f"\nFinal loss is {Loss}") 
# Plotting the training history (losses)
figure(figsize = (13, 5))
ion()
plot(Epochs, listLosses, "red")
title("Training Evaluation")
ylabel("Mean Cross-entropy Train Loss")
xlabel("Epochs")
savefig(Title, dpi = DPI)
show()
