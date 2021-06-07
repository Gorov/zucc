## Source code for our co-matching model
```
run_comatch_bert_sas_jecc.ipynb
```

## Source code for the JECC data generation
To generate our JECC from scratch, first generate the valid actions for each walkthrough step (this step takes time):
```
get_walkthrough_task_valid_action.ipynb
```

See the following code for generating the json files in the data directory.
```
get_sas_task_data.ipynb
get_walkthrough_task_ssa.ipynb
```

## Set up interactive fiction games simulator
The code base is built for Jericho 2.4.3 and 2.4.2. To install Jericho 2.4.2, run: 
```
pip install jericho==2.4.2
```
The game roms can be downloaded following the instructions from the Jericho codebase. Our code requires to put the roms in the [../roms/jericho-game-suite/](../roms/jericho-game-suite/) folder. 


## License
MIT License 
