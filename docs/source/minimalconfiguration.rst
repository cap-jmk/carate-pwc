Minimal Configuration
=====================

This section introduces you to the configuration files. The configuration is rather rigid. 
It is intended that you really think about what you are doing, otherwise you might not do anything after all. 

Don't be like the average human: 

..  tip::

    An average human looks without seeing, listens without hearing, touches without feeling, eats without tasting, moves without physical awareness, inhales without awareness of odour or fragrance, and talks without thinking. - Leonardo da Vinci


**In our modern times we study without learning, and experiment without doing research. The technology
is not useful if you can't build it yourself.**

=========================================
Configuration example for Classification
=========================================

Configurations can be stored as a `.py` file or passed via json. For an example of a configuration 
with JSON please refer to the notebook tutorials.

.. literalinclude:: MCF.py 
    :language: Python

=========================================
Configuration example for Regression
=========================================

For an exmaple of a regression see the configuration file of ALCHEMY

.. literalinclude:: ALCHEMY.py 
    :language: Python

=========================================
Configuration with JSON
=========================================

It might come in handy to use JSON for starting a run. See below for an example json

.. code-block:: json

    {
        "dataset_name" : "PROTEINS",
        "num_classes" : 2,
        "num_features" : 3,
        "model" : "cgc_classification",
        "evaluation" : "classification",
        "optimizer" : "adams",  # defaults to adams optimizer
        "net_dimension" : 364,
        "learning_rate" : 0.0005,
        "dataset_save_path" : "./data",
        "test_ratio" : 20,
        "batch_size" : 64,
        "shuffle" : True,
        "num_epoch" : 10,
        "num_cv" : 1,
        "result_save_dir" : "./PROTEINS_20",
        "data_set" : "StandardTUD",
        "model_save_freq" : 30, 
        "device": "cpu",
        "override": True, 
    }

=========================================
Starting a run from a Jupyter Notebook
=========================================

I recommend to use the JSON mode from a Jupyter notebook. For example you can run 

.. literalinclude:: JSON_start.py 
    :language: Python


=========================================
Starting a run from a config file
=========================================

I recommend to use the JSON mode from a Jupyter notebook. For example you can run 

.. literalinclude:: MCF_without_cli.py 
    :language: Python