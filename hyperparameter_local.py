class HyperparametersLocal:
    ''' hyperparameters '''
    # data
    src1 = 'chr.src'
    src2 = 'bpe.src'
    tgt = 'bpe.tgt'
    path = "./Transliteration/MS_Ch_En/"
    # trainig data
    src1_train = path + 'trn.' + src1 +'.txt'
    src2_train = path + 'trn.' + src2 +'.txt'
    tgt_train = path + 'trn.' + tgt +'.txt'
    # test data
    src1_test = path + 'dev.' + src1 +'.txt'
    src2_test = path + 'dev.' + src2 +'.txt'
    tgt_test = path + 'dev.' + tgt +'.txt'

    prep = path + 'prepMS_translit'

    # training parameters
    logdir = 'log_MS_Ch-En_bpe'
    avg_model = logdir + '_avg'
    result_dir = 'result'
    # logdir = 'log_APE2018_joint'
