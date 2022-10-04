import parse_args
import json
import numpy as np
import pandas as pd
from utils import basics
import glob


def train(model, opt):
    for epoch in range(opt['total_epochs']):
        ifbreak = model.train(epoch)
        if ifbreak:
            break
     
    # record val metrics for hyperparameter selection
    pred_df = model.record_val()
    return pred_df
    

if __name__ == '__main__':
    
    opt, wandb = parse_args.collect_args()
    if not opt['test_mode']:
        
        random_seeds = np.random.choice(range(100), size = 3, replace=False).tolist()
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()
        print('Random seed: ', random_seeds)
        for random_seed in random_seeds:
            opt['random_seed'] = random_seed
            model = basics.get_model(opt, wandb)
            pred_df = train(model, opt)
            val_df = pd.concat([val_df, pred_df])
            
            pred_df = model.test()
            test_df = pd.concat([test_df, pred_df])
            
        stat_val = basics.avg_eval(val_df, opt, 'val')
        stat_test = basics.avg_eval(test_df, opt, 'test')
        model.log_wandb(stat_val.to_dict())
        model.log_wandb(stat_test.to_dict())        
    else:
        
        if opt['cross_testing']:
            
            test_df = pd.DataFrame()
            method_model_path = opt['cross_testing_model_path']
            model_paths = glob.glob(method_model_path + '/cross_domain_*.pth')
            for model_path in model_paths:
                opt['cross_testing_model_path_single'] = model_path
                model = basics.get_model(opt, wandb)
                pred_df = model.test()
                
                test_df = pd.concat([test_df, pred_df])
            stat_test = basics.avg_eval(test_df, opt, 'cross_testing')
            
            model.log_wandb(stat_test.to_dict())