from helpers import prepare_batches, get_target_tensor
import torch
import torch.nn as nn
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
        training_batches = prepare_batches(training_data, batch_size, model.n_chars)
        for input_sequences, target_sequences in training_batches:
            #hx = model.init_hidden(batch_size)
            hx = None
            sequence_lengths = [len(sequence) for sequence in input_sequences]
            y_hat, hx = model(input_sequences, hx, sequence_lengths)
            
            y  = get_target_tensor(target_sequences,
                    sequence_lengths)[0]
            loss = loss_function(y_hat, y)
            loss.backward()
            #consider clipping grad norm
            optimizer.step()
            model.zero_grad()

        if (e + 1) % evaluate_per == 0:
            hx = model.init_hidden(batch_size)

            model.eval()
            validation_batches = prepare_batches(validation_data,
                    batch_size, model.n_chars)
            #get loss per batch
            val_losses = []
            n_batches = 0
            for input_sequences, target_sequences in validation_batches:

                sequence_lengths = [len(sequence) for sequence in input_sequences]
                y_hat, hx = model(input_sequences, hx, sequence_lengths)

                y = get_target_tensor(target_sequences,
                        sequence_lengths)[0]

                val_loss = loss_function(y_hat, y)
                val_losses.append(val_loss.item())
                n_batches += 1
            mean_val_loss = sum(val_losses) / n_batches

            model.train()
            print(f"epoch: {e+1}/{epochs}")
            print(f"training loss: {loss.item():.2f}")
            print(f"validation loss: {mean_val_loss:.2f}")

        #TODO
        #exception for keyboard interrupt
        #anneal learning rate if no improvement has been observed?
        #save model with best validation loss


