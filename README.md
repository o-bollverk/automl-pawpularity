# AutoML course project for kaggle competition 'pawpularity'

https://www.kaggle.com/c/petfinder-pawpularity-score

Instructions and order of running notebooks:
1. Download data from kaggle: https://www.kaggle.com/c/petfinder-pawpularity-score/data
2. Extract the downloaded data into the folder "input"
3. Perform feature engineering:  **Feature Engineering.ipynb** 
4. Calculate baseline; then use hyperopt for hyper-parameter optimization: **baseline.ipynb**
5. Train resnet with hyperopt: **train_resnet_with_hyperopt.ipynb**
6. Use the optimal parameters from 5. to train resnet with the same parameters for a longer time (10 epochs): **resnet_with_optimal_params.ipynb**. Here, the final RMSE is reported


