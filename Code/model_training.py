import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from trainer import TwoStageTrainer,SequenceTrainer
from revisedtrainer import TwoStageTrainer,SequenceTrainer
from newtrainer import SingleStageTrainer

from evaluation import InteractionEvaluator, evaluate_model_performance
from losses import DynamicInteractionLoss, TwoStageLoss

#import warnings
#warnings.filterwarnings('ignore')

def elec_training(model, model_name, train_loader, val_loader, test_loader,scaler, args, device, exp_id):
    save_path = os.path.join(args.save_dirs, args.dataset, 'exp_' + str(exp_id))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()
        df_log_test = pd.DataFrame()
    print(f'Experiment: {exp_id}')
    if model_name=='Dynamic':
        start_time = time.time()
        trainer = SingleStageTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            scaler=scaler)

        print("开始训练...")
        trainer.train(
            epochs=args.stage_a_epochs,
            )
        trainer.plot_training_history(save_path,model_name)
        test_results = trainer.evaluate()
        df_log_test.loc[0, 'mse_test'] = test_results['mse']
        df_log_test.loc[0, 'rmse_test'] = test_results['rmse']
        df_log_test.loc[0, 'mae_test'] = test_results['mae']

        df_log_test.to_csv(os.path.join(save_path, model_name + '_results.csv'))
        np.save(os.path.join(save_path, 'predictions.npy'), test_results['predictions'])
        np.save(os.path.join(save_path, 'main_effects.npy'), test_results['main_effects'])
        np.save(os.path.join(save_path, 'interactions.npy'), test_results['interactions'])
        np.save(os.path.join(save_path, 'contrib_main.npy'), test_results['contrib_main'])
        np.save(os.path.join(save_path, 'contrib_int.npy'), test_results['contrib_int_'])
        np.save(os.path.join(save_path, 'time_weight.npy'), test_results['time_weight'])



        running_time = time.time() - start_time
        print("running time: {:.2f}".format(running_time))
    else:
        start_time = time.time()
        trainer = SequenceTrainer(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            save_dir=save_path,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            scaler=scaler)

        trainer.train(
            )
        #trainer.plot_training_history(save_path, model_name)
        test_results = trainer.evaluate()
        df_log_test.loc[0, 'mse_test'] = test_results['mse']
        df_log_test.loc[0, 'rmse_test'] = test_results['rmse']
        df_log_test.loc[0, 'mae_test'] = test_results['mae']

        df_log_test.to_csv(os.path.join(save_path, model_name + '_results.csv'))
        np.save(os.path.join(save_path, 'predictions.npy'), test_results['predictions'])

        running_time = time.time() - start_time
        print("running time: {:.2f}".format(running_time))














