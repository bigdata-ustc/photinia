#!/usr/bin/env python3

"""
@author: xi
@since: 2016-11-11
"""

import os
import pickle
import re
import shutil

import gridfs
import numpy as np
import pymongo

from . import ops
from . import widgets


class DataSource(object):
    """DataSource
    """

    def next_batch(self, size=0):
        """Get a batch of data.
        
        :param size: Batch size. Default is zero, which means extract all data.
        :return: Tuple of np.array.
        """
        raise NotImplementedError()


class Dataset(DataSource):
    """Dataset
    """

    # “有一种想见不敢见的伤痛，有一种爱还埋藏在我心中，我只能把你放在我的心中”
    # 她问我：“你想见谁啊？”

    def __init__(self,
                 *data,
                 dtype=None):
        """Construct a dataset.
        
        :param data: Tuple of list, np.array or any iterable objects.
        :param dtype: Data type.
        """
        self._num_comp = len(data)
        if self._num_comp == 0:
            raise ValueError('At least 1 data object should be given.')
        self._data = [np.array(mat, dtype=dtype) for mat in data]
        size = None
        for mat in self._data:
            if size is None:
                size = len(mat)
                continue
            if len(mat) != size:
                raise ValueError('All data components must have the same size.')
        self._size = size
        self._start = 0
        self._loop = 0

    @property
    def size(self):
        return self._size

    @property
    def start(self):
        return self._start

    @property
    def loop(self):
        return self._loop

    def next_batch(self, size=0):
        batch = self._next_batch(size)
        if size == 0:
            return batch
        real_size = len(batch[0])
        while real_size < size:
            batch1 = self._next_batch(size - real_size)
            batch = tuple(np.concatenate((batch[i], batch1[i]), 0) for i in range(self._num_comp))
            real_size = len(batch[0])
        return batch

    def _next_batch(self, size=0):
        if size <= 0:
            return self.all()
        if self._start == 0 and self._loop != 0:
            self.shuffle()
        end = self._start + size
        if end < self._size:
            batch = tuple(self._data[i][self._start:end].copy() for i in range(self._num_comp))
            self._start += size
        else:
            batch = tuple(self._data[i][self._start:].copy() for i in range(self._num_comp))
            self._start = 0
            self._loop += 1
        return batch

    def shuffle(self, num=3):
        perm = np.arange(self._size)
        for _ in range(num):
            np.random.shuffle(perm)
        for i in range(self._num_comp):
            self._data[i] = self._data[i][perm]
        return self

    def all(self):
        return self._data


class MongoSource(DataSource):
    """MongoDB data source
    """

    def __init__(self,
                 host,
                 db_name,
                 coll_name,
                 fields):
        #
        self._host = host
        self._db_name = db_name
        self._coll_name = coll_name
        self._fields = fields.copy()
        self._conn = pymongo.MongoClient(self._host)
        self._db = self._conn[db_name]
        self._coll = self._db[coll_name]
        super(MongoSource, self).__init__()

    def next_batch(self, size=0):
        query = self._coll.aggregate([{'$sample': {'size': size}}])
        batch = tuple([] for _ in self._fields)
        for doc in query:
            for index, field in enumerate(self._fields):
                batch[index].append(doc[field])
        return batch

    def close(self):
        self._conn.close()


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


class Trainable(widgets.Widget):
    """Trainable
    """

    # Give me all you got
    # Don't hold nothing back
    # Promise you you're gonna need it
    # Take your best shot I'll just give it back
    # I'm on the winning side we won't be defeated
    # I'm not afraid to die cause life is a battle
    # And fighting me it'll be like fighting shadows

    def __init__(self,
                 name,
                 session,
                 build=True):
        if session is None:
            raise ValueError('Invalid session.')
        self._session = session
        self._slots = {}
        super(Trainable, self).__init__(name, build)

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
                  outputs=None,
                  inputs=None,
                  givens=None,
                  updates=None):
        slot = Slot(
            session=self._session,
            outputs=outputs,
            inputs=inputs,
            updates=updates,
            givens=givens
        )
        self._slots[name] = slot

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


class ModelDumper(object):
    """ModelDumper
    """

    # 你们得救在乎回归安息
    # 你们得力在乎平静安稳

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


if __name__ == '__main__':
    exit()
