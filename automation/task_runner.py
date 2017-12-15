from __future__ import print_function

import logging
import os
import shutil
import sys
import time
from io import StringIO
from multiprocessing import Pool
from functools import partial
from keras import Sequential
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from automation.layers import LayerFactory
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from utils.general import generate_unqiue_file_name
from train import prep
from models import emotion_recognition_cnn
from train import trainer
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)


def myprint(log_file,s):
    print(s, file=log_file)



def run_configuration(layers, log_file,task_file):
    LOGGER.info("Running the configuration")
    X,y = prep.load_faces_dataset()
    X,y, y_orig, class_weight = prep.prepare_data(X, y)
    model = Sequential()

    for layer in layers:
        print('   ', layer, file=log_file)
        model.add(layer.construct())
    myprint_partial = partial(myprint,log_file)
    model.summary(print_fn=myprint_partial)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # Training
    trainer_impl = trainer.Trainer(X, y, y_orig)
    model = trainer_impl.train(model,task_file)
    trainer_impl.evaluate(model)
    trainer_impl.predict(model)
    trainer_impl.export(model)

    return log_file


def run_task(task_file):
    LOGGER.info("Starting tasks {}".format(task_file))
    path_log_file = os.path.basename(task_file)
    log_file = generate_unqiue_file_name(path_log_file, 'log')
    log_file = os.path.join('task_results', log_file)
    layers_fact = LayerFactory()
    with open(task_file, 'r') as f:
        results = []
        layers = []
        for line in f:
            if line.startswith('--run'):
                if len(layers) == 0:
                    continue
                try:
                    string_logger = StringIO()
                    results.append(worker_pool.apply_async(run_configuration, (layers, string_logger,log_file)))
                except Exception as e:
                    logging.exception(e)
                    layers = []
            elif line.startswith('#'):
                # skip comments
                continue
            elif line.strip():
                # assume that a non blank line is a feature extractor
                layers.append(layers_fact.new_from_string(line))
        string_logger = StringIO()
        results.append(worker_pool.apply_async(run_configuration, (layers, string_logger,log_file)))

    with open(log_file, 'w+') as lf:
        for result in results:
            result.wait()
            if result.successful():
                string_buffer = result.get()
                string_buffer.seek(0)
                shutil.copyfileobj(string_buffer, lf, -1)
                string_buffer.close()
                lf.flush()
                LOGGER.info('configuration written to file.')
            else:
                LOGGER.error("Error in configuration.")
                try:
                    result.get()
                except Exception as e:
                    LOGGER.exception(e)
    LOGGER.info("Task {} completed".format(task_file))
    os.rename(task_file, "done/"+task_file)

class TaskDirEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.run'):
            try:
                LOGGER.info('New tasks file detected.')
                run_task(event.src_path)
            except Exception as e:
                logging.exception(e)


def monitor_task_dir(task_dir='tasks'):
    observer = Observer()
    event_handler = TaskDirEventHandler()
    observer.schedule(event_handler, task_dir, recursive=True)
    LOGGER.info('Starting to monitor the tasks directory')
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':

    worker_pool = Pool(4)
    monitor_task_dir('tasks')
