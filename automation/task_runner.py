from __future__ import print_function

import logging
import os
import time
from functools import partial
from multiprocessing import Pool

from keras import Sequential
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from automation.layers import LayerFactory
from train import prep
from train import trainer
from utils.general import generate_unqiue_folder_name, make_dir


def model_printer(log_file, s):
    with open(log_file, 'a') as f:
        print(s, file=f)


def run_configuration(layers, log_folder, datasets):
    logging.info("Running the configuration")

    # Build the model
    model = Sequential()

    for layer in layers:
        model.add(layer.construct())
    model.summary(print_fn=partial(model_printer, log_folder + "summary.log"))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    for X, y, class_weight, name, setting in datasets:
        print("Processing new dataset ({}) with dimensions:".format(name))
        print("X: {}".format(X.shape))
        print("y: {}".format(y.shape))
        print("On the fly augmentation: {}".format(setting))
        trainer_impl = trainer.Trainer(X, y, setting)
        new_log_folder = os.path.join(log_folder, name) + "/"
        make_dir(new_log_folder)
        model, metrics = trainer_impl.train(model, new_log_folder)
        trainer_impl.evaluate(model)
        # trainer_impl.export(model, new_log_folder)
        print("We are going to continue")


def run_task(task_file):
    logging.info("Starting task {}".format(task_file))
    task_file_name = os.path.basename(task_file)
    log_folder = generate_unqiue_folder_name(task_file_name)
    log_folder = os.path.join('task_results', log_folder)
    make_dir(log_folder)
    layers_fact = LayerFactory()

    with open(task_file, 'r') as f:
        layers = []
        for line in f:
            if line.startswith('#'):
                continue
            elif line.strip():
                layers.append(layers_fact.new_from_string(line))
        worker_pool.apply(run_configuration, (layers, log_folder, datasets))
        logging.info("Task {} completed".format(task_file_name))
        os.rename(task_file, os.path.join("tasks", "done", task_file_name))


class TaskDirEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.run'):
            try:
                logging.info('New tasks file detected.')
                run_task(event.src_path)
            except Exception as e:
                logging.exception(e)


def monitor_task_dir(task_dir='tasks'):
    observer = Observer()
    event_handler = TaskDirEventHandler()
    observer.schedule(event_handler, task_dir, recursive=False)
    logging.info('Starting to monitor the tasks directory')
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':
    worker_pool = Pool(1)

    datasets = []
    # load the different datasets
    X, y = prep.unpickle_faces_dataset()
    X, y, class_weight = prep.prepare_data(X, y)
    datasets.append((X, y, class_weight, "augmented", False))

    X, y = prep.load_faces_dataset()
    X, y, class_weight = prep.prepare_data(X, y)
    datasets.append((X, y, class_weight, "non_augmented", False))
    datasets.append((X, y, class_weight, "auto_augment", True))

    monitor_task_dir('tasks')
