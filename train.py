from helpers import prepare_batches, get_target_tensor
import torch
import torch.nn as nn
import time
import pdb

def train(model, training_data, validation_data, 
        epochs, lr, evaluate_per, batch_size):

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    #this is equivalent to a (log) softmax activation layer + negative log likelihood
    loss_function = nn.CrossEntropyLoss()
    
    #I don't understand what's going on with the "device" variable
    #particularly when transferring this to a GPU
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    for e in range(epochs):
        start_time = time.time()
        training_batches = prepare_batches(training_data, batch_size)
        hx = None

        for input_sequences, target_sequences in training_batches:
            #hx = model.init_hidden(batch_size)
            #hx = None
            sequence_lengths = [len(sequence) for sequence in input_sequences]
            y_hat, hx = model(input_sequences, hx, sequence_lengths)
            
            y  = get_target_tensor(target_sequences,
                    sequence_lengths)[0]
            loss = loss_function(y_hat, y)

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

        if (e + 1) % evaluate_per == 0:
            #hx = model.init_hidden(batch_size)

            model.eval()
            validation_batches = prepare_batches(validation_data,
                    batch_size)
            #get loss per batch
            val_loss = 0
            n_batches = 0
            for input_sequences, target_sequences in validation_batches:

                sequence_lengths = [len(sequence) for sequence in input_sequences]
                y_hat, hx = model(input_sequences, hx, sequence_lengths)

                y = get_target_tensor(target_sequences,
                        sequence_lengths)[0]

                loss = loss_function(y_hat, y)
                val_loss += loss.item()
                n_batches += 1

            model.train()
            print(f"validation loss: {val_loss / n_batches:.2f}")

        #TODO
        #exception for keyboard interrupt
        #anneal learning rate if no improvement has been observed?
        #save model with best validation loss


