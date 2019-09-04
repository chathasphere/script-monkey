from helpers import prepare_batches, get_target_tensor
import torch
import torch.nn as nn
import time
from random import shuffle
import pdb

def train(model, training_data, validation_data, 
        epochs, lr, evaluate_per, batch_size):

    model.train() #short hand to begin tracking the gradient
    optimizer = torch.optim.Adam(model.parameters(), lr = lr) #i don't know...? gradient descent equiv
    #this is equivalent to a log softmax activation layer + negative log likelihood
    loss_function = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        #device = torch.device("cuda")
        model.cuda()
        print("GPU is available")
    else:
        #device = torch.device("cpu")
        print("GPU not available, CPU used")

    for e in range(epochs):
        start_time = time.time()
        training_batches = prepare_batches(training_data, batch_size) #returning batches of a given size
        # input and target sequences, latter one time step ahead

        hx = None #this is the hidden state, None means: initializes to zeros
        # hidden state is the "internal representation" of the sequence

        for input_sequences, target_sequences in training_batches:
            
            #skip batches that are undersized
            if len(input_sequences) != batch_size:
                continue
            
            y_hat, hx = model(input_sequences, hx)
            
            y  = get_target_tensor(target_sequences)
                    
            loss = loss_function(y_hat.flatten(0,1), y)

            #don't want to be backpropagating through every timestep, so hidden state
            #is detached from the graph
            hx = tuple(h.detach() for h in hx)
            #clear old gradients from previous step
            model.zero_grad()
            #compute derivative of loss w/r/t parameters
            loss.backward()
            #consider clipping grad norm
            #optimizer takes a step based on gradient
            optimizer.step()
            training_loss = loss.item()

        print(f"epoch: {e+1}/{epochs} | time: {time.time() - start_time:.0f}s")
        print(f"training loss: {training_loss :.2f}")
        shuffle(training_data)

        if (e + 1) % evaluate_per == 0:
            
            #deactivate backprop for evaluation
            model.eval()
            validation_batches = prepare_batches(validation_data,
                    batch_size)
            #get loss per batch
            val_loss = 0
            n_batches = 0
            for input_sequences, target_sequences in validation_batches:

                if len(input_sequences) != batch_size:
                    continue
                sequence_lengths = [len(sequence) for sequence in input_sequences]
                y_hat, hx = model(input_sequences, hx)

                y = get_target_tensor(target_sequences)

                loss = loss_function(y_hat.flatten(0,1), y)
                val_loss += loss.item()
                n_batches += 1

            model.train()
            print(f"validation loss: {val_loss / n_batches:.2f}")
            shuffle(validation_data)


        #TODO
        #exception for keyboard interrupt
        #anneal learning rate if no improvement has been observed?
        #save model with best validation loss


