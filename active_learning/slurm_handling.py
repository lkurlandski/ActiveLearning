



def move_output(experiment_parameters, user_path, job_id_list):
    
    source_dir = config.node_path

    for job_id in job_id_list:
        shutil.move(source_dir + job_id, user_path)
    
    experiment_parameters["output_root"] = user_path