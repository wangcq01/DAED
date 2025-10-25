
def config_elec(model_name, args):
    params = {}
    # some common params for all models
    params['input_dim'] = args.input_dim
    params['hidden_dim'] = args.hidden_dim
    params['num_layers'] = args.num_layers
    params['nhead'] = args.nhead
    params['r_factor'] = args.r_factor
    params['dropout'] = args.dropout
    params['prediction_horizon'] = args.prediction_horizon
    params['batch_size'] = args.batch_size
    params['lr'] = args.lr
    params['weight_decay'] = args.weight_decay
    params['feature_dimensionality'] = args.feature_dimensionality
    params['T']=args.T


    if model_name in ['Dynamic']:
        params['use_bilinear'] = args.use_bilinear
        params['stage_a_epochs'] = args.stage_a_epochs
        params['stage_b_epochs'] = args.stage_b_epochs
        params['lambda_sparsity'] = args.lambda_sparsity
        params['lambda_smooth'] = args.lambda_smooth

    elif model_name == 'Transformer':
        pass
    elif model_name == 'Informer':
        pass
    elif model_name == 'Autoformer':
        pass

    elif model_name == 'GRU':
             pass

    elif model_name == 'Retain':
        params['intputDimSize'] = args.input_dim
        params['embDimSize'] = 256
        params['alphaHiddenDimSize']=128
        params['betaHiddenDimSize'] = 128
        params['outputDimSize']=args.output_dim
        params['keep_prob']=1.0

    elif model_name == 'LSTM':
        pass
    else:
        raise ModuleNotFoundError

    return params

def config(model_name, args):
    if args.dataset == 'electricity':
        params = config_elec(model_name, args)
    elif args.dataset == 'synthetic':
        params = config_elec(model_name, args)
    elif args.dataset == 'weather':
        params = config_elec(model_name, args)

    return params
