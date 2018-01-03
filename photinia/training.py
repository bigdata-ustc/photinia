#!/usr/bin/env python3

"""
@author: xi
@since: 2016-11-11
"""

import collections
import datetime as dt
import os
import pickle
import re
import shutil

import gridfs
import numpy as np
import pymongo
import tensorflow as tf

from . import config
from . import ops
from . import widgets
from . import data


class Slot(object):
    """Slot
    """

    # We humans are more concerned with having than with being.

    def __init__(self,
                 session,
                 outputs=None,
                 inputs=None,
                 updates=None,
                 givens=None):
        #
        # Session.
        if session is None:
            raise ValueError('Invalid session.')
        self._session = session
        #
        # Outputs.
        if outputs is None:
            outputs = ()
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
            self._single_output = True
        else:
            self._single_output = False
        self._outputs = outputs
        self._output_size = len(outputs)
        #
        # Inputs.
        if inputs is None:
            inputs = ()
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        self._inputs = inputs
        #
        # Updates.
        if updates is None:
            updates = ()
        if not isinstance(updates, (tuple, list)):
            updates = (updates,)
        self._updates = updates
        #
        # Givens.
        if givens is None:
            givens = {}
        if not isinstance(givens, dict):
            raise ValueError('Givens must be dict.')
        self._givens = givens
        #
        self._feed_dict = givens.copy()
        self._runnable = outputs + updates
        if len(self._runnable) == 0:
            raise ValueError('At least one output or update should be set.')

    @property
    def outputs(self):
        return self._outputs if isinstance(self._outputs, (tuple, list)) else (self._outputs,)

    @property
    def output(self):
        if isinstance(self._outputs, (tuple, list)):
            assert len(self._outputs) == 1
            return self._outputs[0]
        else:
            return self._outputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def input(self):
        assert len(self._inputs) == 1
        return self._inputs[0]

    @property
    def updates(self):
        return self._updates

    @property
    def update(self):
        assert len(self._updates) == 1
        return self._updates[0]

    @property
    def givens(self):
        return self._givens

    def __call__(self, *args, **kwargs):
        #
        # Check input length.
        if len(args) != len(self._inputs):
            print(len(args), len(self._inputs))
            raise ValueError('The count of parameters is not match the inputs.')
        #
        # Run graph.
        for index, placeholder in enumerate(self._inputs):
            self._feed_dict[placeholder] = args[index]
        outputs = self._session.run(self._runnable, feed_dict=self._feed_dict)
        if self._output_size != 0:
            if self._single_output:
                return outputs[0]
            else:
                return outputs[:self._output_size]
        return None


class Trainer(widgets.Widget):
    """Trainer
    """

    def __init__(self,
                 name,
                 session=None,
                 build=True):
        if session is None:
            tfconfig = tf.ConfigProto()
            tfconfig.gpu_options.allow_growth = True
            self._session = tf.Session(config=tfconfig)
            self._session_provided = False
        else:
            if not isinstance(session, tf.Session):
                raise ValueError('session should be tf.Session.')
            self._session = session
            self._session_provided = True
        self._slots = {}
        self._predict_slot = None
        self._fitters = []
        super(Trainer, self).__init__(name, build)

    def __del__(self):
        if not self._session_provided and self._session is not None:
            self._session.close()

    def _build(self):
        """Build the model.
        Abstract method.
        All subclass must implement this method.

        There are at least two tasks to be done in this method:
        1) Construct the model's graph structure with TF.
        2) Define and add slots for training, evaluation and prediction.
        """
        raise NotImplementedError()

    def _setup(self, *args, **kwargs):
        pass

    def _add_slot(self, name,
                  inputs=None,
                  outputs=None,
                  givens=None,
                  updates=None):
        if name in self._slots:
            raise ValueError('Slot {} exists.'.format(name))
        slot = Slot(
            session=self._session,
            outputs=outputs,
            inputs=inputs,
            updates=updates,
            givens=givens
        )
        self._slots[name] = slot

    def _add_train_slot(self, inputs=None, outputs=None, givens=None, updates=None):
        self._add_slot(config.NAME_TRAIN_SLOT, inputs, outputs, givens, updates)

    def _add_validate_slot(self, inputs=None, outputs=None, givens=None, updates=None):
        self._add_slot(config.NAME_VALID_SLOT, inputs, outputs, givens, updates)

    def _add_predict_slot(self, inputs=None, outputs=None, givens=None, updates=None):
        self._add_slot(config.NAME_PREDICT_SLOT, inputs, outputs, givens, updates)

    def get_slot(self, name):
        return self._slots[name] if name in self._slots else None

    @property
    def parameters(self):
        var_list = self.trainable_variables()
        param_dict = {var.name: self._session.run(var) for var in var_list}
        return param_dict

    @parameters.setter
    def parameters(self, param_dict):
        var_list = self.trainable_variables()
        var_dict = {var.name: var for var in var_list}
        for name, value in param_dict.items():
            if name not in var_dict:
                print('Parameter {} is not in this model.'.format(name))
                continue
            var = var_dict[name]
            var.load(value, session=self._session)

    def initialize_global_variables(self):
        self._session.run(tf.global_variables_initializer())

    def fit(self, max_loop=10000):
        """Train the model to fit the given dataset.

        :param max_loop: The number of max loop. Default is 10000.
            Here, "a loop" means train the model with one batch of data.
        """
        context = {
            config.CONTEXT_TRAINER: self,
            config.CONTEXT_MAX_LOOP: max_loop
        }
        for i in range(1, max_loop + 1):
            context[config.CONTEXT_LOOP] = i
            for fitter in self._fitters:
                try:
                    fitter.fit(i, max_loop, context)
                except FitterInterrupt:
                    break

    def add_fitter(self, fitter):
        self._fitters.append(fitter)

    def add_data_fitter(self,
                        data_source,
                        batch_size,
                        slot_name,
                        interval=1,
                        count=1):
        self.add_fitter(DataFitter(data_source, batch_size, self, slot_name, interval, count))

    def add_data_trainer(self,
                         data_source,
                         batch_size,
                         interval=1,
                         count=1):
        self.add_fitter(DataFitter(data_source, batch_size, self, config.NAME_TRAIN_SLOT, interval, count))

    def add_data_validator(self,
                           data_source,
                           batch_size,
                           interval=1,
                           count=1):
        self.add_fitter(Validator(data_source, batch_size, self, config.NAME_VALID_SLOT, interval, count))

    def add_screen_logger(self,
                          log_attr,
                          value_names=('loss',),
                          interval=1,
                          count=1):
        self.add_fitter(ScreenLogger(log_attr, value_names, interval, count))

    def predict(self, data_batch):
        if self._predict_slot is None:
            if config.NAME_PREDICT_SLOT not in self._slots:
                raise RuntimeError('No predict slot defined.')
            self._predict_slot = self._slots[config.NAME_PREDICT_SLOT]
        return self._predict_slot(*data_batch)


class FitterInterrupt(BaseException):
    """Fit process is interrupt by one of the fitters.
    """

    def __init__(self, *args, **kwargs):
        pass


class Fitter(object):
    """Fitter
    """

    def __init__(self,
                 interval=1,
                 count=1):
        self._interval = interval
        self._count = count

    def fit(self, i, max_loop, context):
        if i % self._interval == 0:
            for _ in range(self._count):
                self._fit(i, max_loop, context)

    def _fit(self, i, max_loop, context):
        raise NotImplementedError()


class DataFitter(Fitter):
    """Data fitter
    """

    def __init__(self,
                 data_source,
                 batch_size,
                 trainer,
                 slot_name,
                 interval=1,
                 count=1):
        super(DataFitter, self).__init__(interval, count)
        if not isinstance(data_source, data.DataSource):
            raise ValueError('data_source should be an instance of training.DataSource.')
        self._ds = data_source
        if batch_size < 0:
            raise ValueError('batch_size should not be negative.')
        self._batch_size = batch_size
        if not isinstance(trainer, Trainer):
            raise ValueError('trainer should be an instance of training.Trainer.')
        self._trainable = trainer
        self._slot_name = slot_name
        self._slot = trainer.get_slot(slot_name)

    def _fit(self, i, max_loop, context):
        data_batch = self._ds.next_batch(self._batch_size)
        ret = self._slot(*data_batch)
        context[self._slot_name] = ret


class Validator(DataFitter):
    """Validator
    """

    def __init__(self,
                 data_source,
                 batch_size,
                 trainer,
                 slot_name,
                 interval=1,
                 count=1):
        super(Validator, self).__init__(
            data_source=data_source,
            batch_size=batch_size,
            trainer=trainer,
            slot_name=slot_name,
            interval=interval,
            count=count
        )

    def _fit(self, i, max_loop, context):
        ret_list = []
        data = self._ds.next_batch(0)
        size = len(data[0])
        batch_size = self._batch_size
        for i in range(1, size // batch_size + 1):
            data_batch = tuple(comp[(i - 1) * batch_size: i * batch_size] for comp in data)
            ret = self._slot(*data_batch)
            if not isinstance(ret, (tuple, list)):
                ret = (ret,)
            ret_list.append(ret)
        last_size = size % batch_size
        if last_size != 0:
            data_batch = tuple(comp[-last_size:] for comp in data)
            ret = self._slot(*data_batch)
            if not isinstance(ret, (tuple, list)):
                ret = (ret,)
            ret_list.append(ret)
        context[self._slot_name] = tuple(comp for comp in np.sum(ret_list, axis=0) / size)


class ScreenLogger(Fitter):
    """Screen logger
    """

    def __init__(self,
                 log_attribute,
                 value_names=('loss',),
                 interval=1,
                 count=1):
        super(ScreenLogger, self).__init__(interval, count)
        self._log_attribute = log_attribute
        self._value_names = value_names

    def _fit(self, i, max_loop, context):
        now = dt.datetime.now()
        print(now.strftime('[%Y-%m-%d %H:%M:%S '), end='')
        # loop = context[config.CONTEXT_LOOP] if config.CONTEXT_LOOP in context else '?'
        # max_loop = context[config.CONTEXT_MAX_LOOP] if config.CONTEXT_MAX_LOOP in context else '?'
        percentage = '%.2f' % (i / max_loop * 100,)
        print('%s/%s|%s%%]' % (str(i), str(max_loop), percentage), end='')
        #
        values = context[self._log_attribute] if self._log_attribute in context else ()
        if not isinstance(values, (list, tuple)):
            values = (values,)
        for i, name in enumerate(self._value_names):
            if i < len(values):
                value = values[i]
                print('\t%s=%f' % (name, value), end='')
            else:
                print('\t%s=?' % (name,), end='')
        print()


class MPIDispatcher(Fitter):
    """MPI Dispatcher

    This class is used for the distributional training of the model. (Based on MPI).
    So, the servers should have one of the MPI implementation (e.g., openmpi, mpich) installed.
    If this fitter is instanced and added to a trainer, the program should be run using the MPI command:

        mpiexec -n {num_processes} python3 {python_file.py}
    """

    def __init__(self,
                 sync_interval=2):
        super(MPIDispatcher, self).__init__(1, 1)
        from mpi4py import MPI
        self._sync_interval = sync_interval
        #
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()
        #
        # This is very important since we should let the processes to use DIFFERENT GPUs of the same server.
        # While, if the processes run on different servers, this can cause problems.
        # TODO: Thus we need to further modify the assign policy to choose the GPU automatically.
        gpu_list = [int(item) for item in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpu = gpu_list[self._rank % len(gpu_list)]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    def _fit(self, i, max_loop, context):
        trainer = context[config.CONTEXT_TRAINER]
        if i == 1:
            self._init_all(trainer)
        elif i % self._sync_interval == 0:
            self._update_all(trainer)

    def _init_all(self, trainer):
        if self._rank == 0:
            self._comm.bcast(trainer.parameters, root=0)
        else:
            trainer.parameters = self._comm.bcast(None, root=0)

    def _update_all(self, trainer):
        if self._rank == 0:
            #
            # Gather parameters from all processes (include the master itself).
            # Compute the mean value for each parameter.
            # Then, broadcast them.
            param_list = self._comm.gather(trainer.parameters, root=0)
            new_params = collections.defaultdict(list)
            for params in param_list:
                for name, value in params.items():
                    new_params[name].append(value)
            new_params = {key: np.mean(value_list, axis=0) for key, value_list in new_params.items()}
            new_params = trainer.parameters = self._comm.bcast(new_params, root=0)
        else:
            self._comm.gather(trainer.parameters, root=0)
            new_params = self._comm.bcast(None, root=0)
        #
        # Update the parameters to the same version for all processes.
        trainer.parameters = new_params


class ModelDumper(object):
    """ModelDumper
    """

    def dump(self, name, model):
        """Dump the model to somewhere (file, DB, ...) using the given name.

        :param name: The output name. (Not the model name. Note that the output is just one instance of the model.)
        :param model: The model to be dumped.
        """
        param_dict = model.parameters
        self._dump(name, param_dict)

    def _dump(self, name, param_dict):
        raise NotImplementedError

    def load(self, name, model, alias_list=None):
        param_dict = self._load(name)
        if alias_list:
            new_dict = {}
            for key, value in param_dict.items():
                for src, dst in alias_list:
                    if not key.startswith(src):
                        continue
                    print(key)
                    if isinstance(dst, widgets.Widget):
                        dst = dst.prefix()
                    key, _ = re.subn('^{}'.format(src), dst, key)
                    new_dict[key] = value
            param_dict = new_dict
        model.parameters = param_dict

    def _load(self, name):
        raise NotImplementedError


class FileDumper(ModelDumper):
    """File Dumper
    """

    def __init__(self,
                 output_dir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self._output_dir = output_dir
        super(FileDumper, self).__init__()

    @property
    def output_dir(self):
        return self._output_dir

    def clear(self):
        if os.path.exists(self._output_dir):
            shutil.rmtree(self._output_dir)
            os.mkdir(self._output_dir)

    def _dump(self, name, param_dict):
        model_file = os.path.join(self._output_dir, name)
        with open(model_file, 'wb') as f:
            pickle.dump(param_dict, f)

    def _load(self, name):
        param_file = os.path.join(self._output_dir, name)
        with open(param_file, 'rb') as f:
            return pickle.load(f)


class TreeDumper(FileDumper):
    """Tree Dumper

    Dump a model into a directory as a tree form.
    For example, a model with parameters {model/h1/b:0, model/h1/w:0} will be dumped in the following form:
    model/
    ....h1/
    ........w.0
    ........b.0
    """

    def __init__(self,
                 output_dir):
        super(TreeDumper, self).__init__(output_dir)

    def _dump(self, name, param_dict):
        model_dir = os.path.join(self._output_dir, name)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)
        for path, value in param_dict.items():
            param_dir, _ = os.path.split(path)
            param_dir = os.path.join(model_dir, param_dir)
            param_file = os.path.join(model_dir, path)
            param_file = TreeDumper._escape(param_file)
            if not os.path.exists(param_dir):
                os.makedirs(param_dir)
            with open(param_file, 'wb') as f:
                pickle.dump(value, f)

    @staticmethod
    def _escape(path):
        path = list(path)
        for i in range(len(path) - 1, -1, -1):
            ch = path[i]
            if ch == os.sep:
                break
            if ch == ':':
                path[i] = '.'
        return ''.join(path)

    def _load(self, name):
        model_dir = os.path.join(self._output_dir, name)
        if not os.path.exists(model_dir):
            raise FileNotFoundError()
        param_dict = {}
        for path in os.listdir(model_dir):
            TreeDumper._load_tree(model_dir, path, param_dict)
        return param_dict

    @staticmethod
    def _load_tree(model_dir, path, param_dict):
        real_path = os.path.join(model_dir, path)
        if os.path.isdir(real_path):
            for subpath in os.listdir(real_path):
                subpath = os.path.join(path, subpath)
                TreeDumper._load_tree(model_dir, subpath, param_dict)
        elif os.path.isfile(real_path):
            path = TreeDumper._unescape(path)
            with open(real_path, 'rb') as f:
                value = pickle.load(f)
                param_dict[path] = value

    @staticmethod
    def _unescape(path):
        path = list(path)
        for i in range(len(path) - 1, -1, -1):
            ch = path[i]
            if ch == os.sep:
                break
            if ch == '.':
                path[i] = ':'
        return ''.join(path)


class MongoDumper(ModelDumper):
    """MongoDB Model Dumper
    """

    def __init__(self, host, db_name, coll='models'):
        self._host = host
        self._db_name = db_name
        self._coll = coll
        super(MongoDumper, self).__init__()

    def clear(self):
        with pymongo.MongoClient(self._host) as conn:
            db = conn[self._db_name]
            coll1 = db[self._coll + '.files']
            coll2 = db[self._coll + '.chunks']
            coll1.remove()
            coll2.remove()

    def _dump(self, name, param_dict, **kwargs):
        with pymongo.MongoClient(self._host) as conn:
            db = conn[self._db_name]
            fs = gridfs.GridFS(db, collection=self._coll)
            if fs.exists(name):
                fs.delete(name)
            with fs.new_file(_id=name, **kwargs) as f:
                pickle.dump(param_dict, f)

    def _load(self, name):
        with pymongo.MongoClient(self._host) as conn:
            db = conn[self._db_name]
            fs = gridfs.GridFS(db, collection=self._coll)
            f = fs.find_one({'_id': name})
            if f is None:
                return None
            with f:
                param_dict = pickle.load(f)
        return param_dict


class OptimizerWrapper(object):
    """OptimizerWrapper
    """

    def __init__(self,
                 optimizer):
        self._optimizer = optimizer

    @property
    def optimizer(self):
        return self._optimizer

    def minimize(self, loss, var_list=None):
        pair_list = self._optimizer.compute_gradients(loss, var_list=var_list)
        pair_list = self._process_gradients(pair_list)
        return self._optimizer.apply_gradients(pair_list)

    def _process_gradients(self, pair_list):
        raise NotImplementedError


class GradientClipping(OptimizerWrapper):
    """GradientClipping
    """

    def __init__(self, optimizer, max_norm):
        self._max_norm = max_norm
        super(GradientClipping, self).__init__(optimizer)

    @property
    def max_norm(self):
        return self._max_norm

    def _process_gradients(self, pair_list):
        pair_list, raw_grad, grad = ops.clip_gradient(pair_list, self._max_norm)
        self._raw_grad_norm = raw_grad
        self._grad_norm = grad
        return pair_list

    @property
    def raw_grad_norm(self):
        return self._raw_grad_norm

    @property
    def grad_norm(self):
        return self._grad_norm
