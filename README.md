DSPipelineProject

	Main Branch
		config —> ./config/config.json —> contains model paths, training params etc
		src/main.py —> Train the model(calls train_model.py)
		src/create_metrics.py --> Create model performance metrics 
	
	feature/eda Branch
		config —> ./config/config.ini —> contains augmentation types, dataset paths etc
		src —> Augmentation code
		
	
	eda/Histogram Branch
		Distribution.ipynb ---> Contains the Pixel distribution code
		Histogram.ipynb —> Contains other Eda code 
