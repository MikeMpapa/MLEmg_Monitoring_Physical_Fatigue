## Original Publication
Papakostas, Michalis, et al. "Physical fatigue detection through EMG wearables and subjective user reports: a machine learning approach towards adaptive rehabilitation." Proceedings of the 12th ACM International Conference on PErvasive Technologies Related to Assistive Environments. ACM, 2019.
#### Bibtec
@inproceedings{papakostas2019physical,
  title={Physical fatigue detection through EMG wearables and subjective user reports: a machine learning approach towards adaptive rehabilitation},
  author={Papakostas, Michalis and Kanal, Varun and Abujelala, Maher and Tsiakas, Konstantinos and Makedon, Fillia},
  booktitle={Proceedings of the 12th ACM International Conference on PErvasive Technologies Related to Assistive Environments},
  pages={475--481},
  year={2019},
  organization={ACM}
}


# EMG_Fatigue_Monitoring
Dependencies: 

[pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)

[Keras](https://keras.io/) - optional for CNN classification

### Run to train & evaluate:
```
python FatigueMonitoring
```
 Edit the code under the main function to run different evaluation scenarios
 
 Processed Dataset used for this study available in the repo
 
 

 Original Dataset available at: https://www.dropbox.com/s/daj4du1ra5zo821/Fatigue%20Data.zip?dl=0

### Data format

1. experiment_comparison folder contains temporal evaluations in the format : c1:timestamp c2:ground_truth c3:predicted

2. Study1_medfilt11_EMG folder contains the processed EMG measurments by a median filter with window size  of 11 samples

3. original_labels_Study1 folder contains the labels provided by the human subjects: 0=NO_FATIGUE, 1=FATIGUE

4. original_times_Study1 folder contains the timestamps of each EEG measurment

5. For questions in the code check the comments 

6. Dataset file format: <user_id>_<exercise_id>_<repetition_id>


## Original Publication
Papakostas, Michalis, et al. "Physical fatigue detection through EMG wearables and subjective user reports: a machine learning approach towards adaptive rehabilitation." Proceedings of the 12th ACM International Conference on PErvasive Technologies Related to Assistive Environments. ACM, 2019.
#### Bibtec
@inproceedings{papakostas2019physical,
  title={Physical fatigue detection through EMG wearables and subjective user reports: a machine learning approach towards adaptive rehabilitation},
  author={Papakostas, Michalis and Kanal, Varun and Abujelala, Maher and Tsiakas, Konstantinos and Makedon, Fillia},
  booktitle={Proceedings of the 12th ACM International Conference on PErvasive Technologies Related to Assistive Environments},
  pages={475--481},
  year={2019},
  organization={ACM}
}
