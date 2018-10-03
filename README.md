# EMG_Fatigue_Monitoring

Dependencies: [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)

Run to train classifier:
```
python EMG_training_new.py <path_to_data> -svm <list_of_params_to_evaluate_classifier> <apply_median_filter: Boolean>
```

Run to visualise raw data: 
```
python EMG_training_new.py <path_to_data> -s
```
