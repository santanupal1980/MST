class Hyperparameters:
    ''' hyperparameters '''

    # training parameters
    batch_size = 32
    lr = 0.01  # need to change it auto adjusted and analying

    # model parameters
    maxlen = 25
    min_cnt = 2
    hidden_units = 256
    num_blocks = 16
    num_epochs = 15
    num_heads = 8
    dropout_rate = 0.3
    sinusoid = False  # F=positional embeddings, T = sinusoid
