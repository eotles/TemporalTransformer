import Hopper
import sqlite_utils
import sql_statements

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics, calibration

import sys
import warnings


dur_fn = "i_duration"

select_cols_sql = sql_statements.select_cols_sql

#TODO: check that model is being masked correctly
#TODO: move graph functions to a util file
#TODO: when asked to fit on an output that has_times=False we change the formulation
# becomes seqeunce input -> single label prediction



def tokenize_and_sample(obs_str, tokenizer, channels=1):
    obs = tokenizer.texts_to_sequences([obs_str])[0]

    if len(obs) <= channels:
        n = channels - len(obs)
        empty_token = tokenizer.texts_to_sequences(['_empty_'])[0]
        res = obs + empty_token*(n)

    else:
        res = obs[:channels]

    return(res)
    

#makes data generation function from feature column data (rows per idx)
def make_data_gen(fn_data):
    return(lambda: iter(fn_data))

@tf.function
def expand(x):
    return tf.expand_dims(x, axis = -1)



class tf_prepper():

    fn_template = "{tn}/{cn}"

    def __init__(self, dbms, fvms=None, partition=None):
        self.dbms = dbms
        self.cur_man = self.dbms.cur_man
        self._init_features(fvms=fvms, partition=None)
        
    @property
    def input_features(self):
        fns = [fn for fn in self.features if fn not in self.ignore_fns]
        return(fns)

    
    def _init_features(self, fvms=None, partition=None):
        if fvms is None:
            fvms = self.dbms.fvms
        self.fvms = fvms
        
        features = []
        config = {}
        fn_lookup = {} #TODO: RM
        
        self.ignore_fns = [Hopper.pk_cn, dur_fn]
        for fn in self.ignore_fns:
            features.append(fn)
            config[fn] = {"cn": None, "fvm": None, "type": "meta"}
        
        for fvm in self.fvms:
            tc = fvm.get_partition_tc("nrm", partition=None)
            tn = fvm.data_table_man.table_config.name
            
            for cn, ctype in zip(tc.feature_names, tc.feature_types):
                new_fn = self.fn_template.format(tn=tn, cn=cn)
                features.append(new_fn)
                config[new_fn] = fvm.nrm_config[cn]
                config[new_fn]["cn"] = cn
                config[new_fn]["fvm"] = fvm
        
        #TODO: test bringing back one level of tabbing
            self.features = features
            self.config = config
            
            
    def _fit_tokenizers(self, partition=None, no_time_channels=1, has_time_hdc_channels=5):
        from_view = "nrm"
        
        for fn, fn_config in self.config.items():
            if fn_config["type"] == "hdc":
                cn = fn_config["cn"]
                fvm = fn_config["fvm"]
                tc = fvm.get_partition_tc(from_view, partition=partition)
                hdc_texts_sql = select_cols_sql.format(cols=cn, tn=tc.name)
                hdc_texts = self.cur_man.execute_fetchall(hdc_texts_sql)
                hdc_texts = [row[0] for row in hdc_texts]
                hdc_texts.append('_empty_')
                hdc_tokenizer = tf.keras.preprocessing.text.Tokenizer(split=',', oov_token='_unknown_', filters="")
                hdc_tokenizer.fit_on_texts(hdc_texts)
                fn_config["tokenizer"] = hdc_tokenizer
                fn_config["channels"] = has_time_hdc_channels if tc.has_times else no_time_channels
                
    
    def __update_base_data(self, idx, st, fn, val, ignore_config=False):
        if not ignore_config:
            ctype = self.config[fn]["type"]
            
            if ctype == "hdc":
                tokenizer = self.config[fn]["tokenizer"]
                channels = self.config[fn]["channels"]
                val = tokenize_and_sample(val, tokenizer, channels=channels)
    
        if val is None:
            val = 0.0
    
        #TODO: investigate whether less mem usage if we use np
        #TODO: somehow getting values larger than duration
        if st is None:
            if type(val) == list:
                self.base_data[idx][fn] = [val]*self.durations[idx]
            else:
                self.base_data[idx][fn] = [[val]]*self.durations[idx]
        else:
            try:
                if type(val) == list:
                    self.base_data[idx][fn][st] = val
                else:
                    self.base_data[idx][fn][st] = [val]
            except:
                msg = "unable to put update base_data with (idx: %s, fn: %s, st: %s, val: %s)... hint duration[idx] is: %s" %(idx, fn, st, val, self.durations[idx])
                warnings.warn(msg)
            
    
    #TODO: input validation
    def fit(self, partition=None, no_time_channels=1, has_time_hdc_channels=5,
            offsets = [], label_fns = [], ignore_fns = []):
            
        self.has_time_hdc_channels = has_time_hdc_channels
        self._fit_tokenizers(partition=partition, no_time_channels=1, has_time_hdc_channels=5)
        
        self.offsets = offsets
        self.label_fns = label_fns
        self.ignore_fns = self.ignore_fns + ignore_fns
    
    
    def _form_base_data(self):
        from_view = "nrm"
        
        windows_sql = select_cols_sql.format(cols="*", tn=Hopper.wn_tn)
        windows = self.cur_man.execute_fetchall(windows_sql)
        self.durations = {idx: et+1 for idx, _, et in windows}
        #NOTE: we are adding 1 to duration - do we need this?
        self.idxs = list(self.durations.keys())
        self.base_data = {idx: {} for idx in self.idxs}
        
        #idx & duration
        for idx, duration in self.durations.items():
            self.__update_base_data(idx, None, Hopper.pk_cn, idx, ignore_config=True)
            self.__update_base_data(idx, None, dur_fn, duration, ignore_config=True)
            
        
        for fvm in self.fvms:
            tc = fvm.view_tc[from_view]
            tn = fvm.data_table_man.table_config.name
            
            #prepare base_data[fn] for reception of new data
            for cn, ctype in zip(tc.feature_names, tc.feature_types):
                fn = self.fn_template.format(tn=tn, cn=cn)
                for idx, duration in self.durations.items():
                    val = "" if ctype == "hdc" else 0.0
                    self.__update_base_data(idx, None, fn, val)
            
            #load data
            table_data_sql = select_cols_sql.format(cols="*", tn=tc.name)
            table_data = self.cur_man.execute_fetchall(table_data_sql)
            
            if tc.has_times:
                for idx, st, et, *row_data in table_data:
                    for cn, val in zip(tc.feature_names, row_data):
                        fn = self.fn_template.format(tn=tn, cn=cn)
                        self.__update_base_data(idx, st, fn, val)
            else:
                for idx, *row_data in table_data:
                    for cn, val in zip(tc.feature_names, row_data):
                        fn = self.fn_template.format(tn=tn, cn=cn)
                        self.__update_base_data(idx, None, fn, val)
    
    
    def transform_to_ds(self, buffer_size=1000, batch_size=64):
        self._form_base_data()
        
        partition_table = self.cur_man.execute_fetchall(select_cols_sql.format(cols="*", tn=Hopper.pr_tn))
        partition_lookup = {idx: partition for idx, partition in partition_table}
        partitions = set(partition_lookup.values())
        
        self.partition_lookup = partition_lookup
        
        #prepare X datasets
        x_dss = {partition: {} for partition in partitions}
        x_padded_shapes = {}
        
        for fn in self.features:
            fn_data = {partition: [] for partition in partitions}
            for idx, idx_data in self.base_data.items():
                partition = partition_lookup[idx]
                fn_data[partition].append(idx_data[fn])
            output_types = tf.float32
            x_padded_shapes[fn] = [None, 1]
            
            if fn == Hopper.pk_cn:
                output_types = tf.string
            
            if fn in self.config:
                fn_config = self.config[fn]
                if fn_config["type"] == "hdc":
                    output_types = tf.int32
                    x_padded_shapes[fn] = [None, fn_config["channels"]]
            
            for partition in partitions:
                data_gen = make_data_gen(fn_data[partition])
                par_fn_ds = tf.data.Dataset.from_generator(data_gen, output_types)
                x_dss[partition][fn] = par_fn_ds
        
        #prepare Y datasets
        #produce labels from offsetting
        label_template = "{fn}/{offset}"
        
        labels = {idx: {} for idx in self.idxs}
        for idx, idx_data in self.base_data.items():
            for fn in self.label_fns:
                labels[idx][fn] = []
                for offset in self.offsets:
                    ln = label_template.format(fn=fn, offset=offset)
                    l = idx_data[fn]
                    o = min(offset, len(l))
                    labels[idx][fn].append(l[o:] + [l[-1]]*o)
        self.labels = labels
        
        y_dss = {partition: [] for partition in partitions}
        y_padded_shapes = []
        for label_fn in self.label_fns:
            label_data = {partition: [] for partition in partitions}
            for idx, idx_data in self.labels.items():
                partition = partition_lookup[idx]
                label_data[partition].append(idx_data[label_fn])
            
            for partition in partitions:
                data_gen = make_data_gen(label_data[partition])
                par_label_ds = tf.data.Dataset.from_generator(data_gen, tf.float32)
                par_label_ds = par_label_ds.map(lambda x: tf.squeeze(x, [-1]))
                par_label_ds = par_label_ds.map(lambda x: tf.transpose(x))
                
                y_dss[partition].append(par_label_ds)
            
            y_padded_shapes.append([None, len(self.offsets)])
            
        
        y_dss = {partition: tuple(par_y_dss)
                 for partition, par_y_dss in y_dss.items()}
        y_padded_shapes = tuple(y_padded_shapes)
        
        #merge
        self.ds = {}
        for partition in partitions:
            par_x_ds = tf.data.Dataset.zip(x_dss[partition])
            par_y_ds = tf.data.Dataset.zip(y_dss[partition])
            par_ds = tf.data.Dataset.zip((par_x_ds, par_y_ds))
            par_ds = par_ds.shuffle(buffer_size=buffer_size)
            par_ds = par_ds.padded_batch(batch_size=batch_size, padded_shapes=(x_padded_shapes, y_padded_shapes))
            self.ds[partition] = par_ds
        
        
        #TODO: should do some garbage collection
        return(self.ds)
        
        
    def build_input_model(self, cat=True):
        inputs = {}
        outputs = {}
        
        for fn in self.input_features:
            fn_config = self.config[fn]
        
            if fn_config["type"] == "hdc":
                channels = fn_config["channels"]
                input_dim = len(fn_config["tokenizer"].word_index)+1
                output_dim = round(input_dim**(0.25)+1)
                emb_name = "emb_%s" %(fn)
                rsh_name = "rsh_%s" %(fn)
                
                in_layer = tf.keras.layers.Input(shape=(None, channels), name=fn)
                emb_layer = tf.keras.layers.Embedding(input_dim, output_dim,
                                                      mask_zero=True,
                                                      name=emb_name)(in_layer)
                out_layer = tf.keras.layers.Reshape([-1, channels*output_dim],
                                                    name=rsh_name)(emb_layer)
            else:
                in_layer = out_layer = tf.keras.layers.Input(shape=(None, 1), name=fn)
            
            inputs[fn] = in_layer
            outputs[fn] = out_layer
                
        input_list = list(inputs.values())
        output_list = list(outputs.values())
        if cat:
            cat_inputs = tf.keras.layers.concatenate(output_list, axis=-1, name="cat_inputs")
            ingestion_model = tf.keras.models.Model(inputs=input_list,
                                                    outputs=cat_inputs,
                                                    name="ingestion")
        else:
            ingestion_model = tf.keras.models.Model(inputs=input_list,
                                                    outputs=output_list,
                                                    name="ingestion")
        
        return(ingestion_model)


    def build_output_layers(self, activation="sigmoid"):
        output_layers = []
        for label_fn in self.label_fns:
            output_layers.append(tf.keras.layers.Dense(units=len(self.offsets),
                                                       activation=activation,
                                                       name="out_%s" %(label_fn)))
        return(output_layers)
       
       
    def build_model(self, middle_layer_list=None, cat=True, activation="sigmoid"):
        input_model = self.build_input_model(cat=cat)
        output_layers = self.build_output_layers(activation=activation)
        
        prev_layer = input_model.output
        if middle_layer_list:
            for curr_layer in middle_layer_list:
                prev_layer = curr_layer(prev_layer)
        
        outputs = []
        for output_layer in output_layers:
            outputs.append(output_layer(prev_layer))
                
        final_model = tf.keras.models.Model(inputs=input_model.inputs,
                                            outputs=tuple(outputs),
                                            name="final")
        return(final_model)
        
    
    def get_specific_XY(self, idx):
        #s_X = {fn: tf.convert_to_tensor([v])
        #       for fn, v in self.base_data[idx].items()}
        s_X = {}
        for fn, v in self.base_data[idx].items():
            if fn == Hopper.pk_cn:
                output_types = tf.string
            elif fn==dur_fn or self.config[fn]["type"]=="hdc":
                output_types = tf.int32
            else:
                output_types = tf.float32
            s_X[fn] = tf.convert_to_tensor([v], dtype=output_types)
        
        
        s_Y = {lb: tf.convert_to_tensor(v)
               for lb, v in self.labels[idx].items()}
        return(s_X, s_Y)
        
    
    def inverse_transform(self, fn, val):
        fn_config = self.config[fn]
        fn_type = fn_config["type"]
        
        if fn_type == "real":
            if val != 0:
                u = fn_config["avg"]
                s = fn_config["var"]**(0.5)
                val = round((val*s)+u, 1)
                return(val)
        
        elif fn_type == "ldc":
            if val != 0:
                val = round(val, 1)
                return(val)
        
        elif fn_type == "hdc":
            if type(val) is np.ndarray:
                val = val.tolist()
            elif type(val) != list:
                val = [val]
            val = fn_config["tokenizer"].sequences_to_texts([val])[0]
            val = val.split()
            val = [i for i in val if i != "_empty_"]
            if len(val) == 0:
                val = None
            else:
                val = ", ".join(val)
            return(val)
        
        return None
        


class entity():

    def __init__(self, idx, tfp, model=None):
        s_X, s_Y = tfp.get_specific_XY(idx)
        
        #human readable history
        hrh = {}
        hrh[Hopper.pk_cn] = s_X[Hopper.pk_cn].numpy()[0][0][0].decode("UTF-8")
        hrh[dur_fn] = s_X[dur_fn].numpy()[0][0][0]

        hrh["characteristics"] = {}
        hrh["samples"] = {i: {} for i in range(hrh[dur_fn])}

        for fn, fn_vals in s_X.items():
            fn_vals = np.squeeze(fn_vals.numpy())

            if fn in tfp.input_features:
                fn_config = tfp.config[fn]
                if fn_config["fvm"].tc.has_times:
                    for i, v in enumerate(fn_vals):
                        v = tfp.inverse_transform(fn, v)
                        if v is not None:
                            hrh["samples"][i][fn] = v
                else:
                    v = tfp.inverse_transform(fn, fn_vals[0])
                    if v is not None:
                        hrh["characteristics"][fn] = v
        
        groundTruth = {}
        trainingLabel = {}
        for lb_i, lb in enumerate(tfp.label_fns):
            gt_fn = s_X[lb].numpy()
            gt_fn = np.reshape(gt_fn, (-1, 1))
            groundTruth[lb] = gt_fn
            
            trainingLabel[lb] = s_Y[lb].numpy()
            if len(tfp.offsets)>1:
                trainingLabel[lb] = np.hstack(trainingLabel[lb])
        
        self.idx = idx
        self.tfp = tfp
        self.model = model
        self.s_X = s_X
        self.s_Y = s_Y
        self.hrh = hrh
        self.groundTruth = groundTruth
        self.trainingLabel = trainingLabel
        
        if model is not None:
            self.predict(model)

        
    def predict(self, model):
        self.model = model
        
        s_Y_hat = model.predict(self.s_X)
        
        predictions = {}
        if len(self.tfp.label_fns)==1:
            predictions[self.tfp.label_fns[0]] = np.hstack(s_Y_hat)
        else:
            for lb_i, lb in enumerate(self.tfp.label_fns):
                predictions[lb] = np.hstack(s_Y_hat[lb_i])
        
        self.predictions = predictions
        

    def print_labels(self):
        
        for lb in self.tfp.label_fns:
            print("{}: {}".format(self.idx, lb))
            
            gt = self.groundTruth[lb]
            tl = self.trainingLabel[lb]

            if self.model is None:
                for _x, _y in zip(gt, tl):
                    print(_x, _y)
            else:
                lb_y_hat = self.predictions[lb].round(2)
                for _x, _y, _y_hat in zip(gt, tl, lb_y_hat):
                    print(_x, _y, _y_hat)
                    
            print()
        
    
    def plot(self):
        for lb in self.tfp.label_fns:
            plt.plot(self.groundTruth[lb], 'k')
            plt.title("".format(lb))
            plt.title("{}".format(self.idx), loc="left")
            plt.title("{}".format(lb), loc="right")
            plt.ylabel("Estimated Probability")
            plt.xlabel("Time")

            if self.model is not None:
                for i_o, o in enumerate(self.tfp.offsets):
                    plt.plot(self.predictions[lb][:, i_o], label=o)
                plt.legend(loc="upper left")
            plt.show()
            
    
#TODO: graph axes should be static
def roc_curves(ys, ps, ns, coalate=True, lb=""):
    plt.figure()
    plt.plot([0, 1], [0, 1], "k:")
    
    label_template = "{} (AUC: {:.2f})"
    
    all_y = []
    all_p = []
    for y, p, n in zip(ys, ps, ns):
        all_y += y
        all_p += p
        fpr, tpr, thresholds = metrics.roc_curve(y, p)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=label_template.format(n, roc_auc))
    
    if coalate:
        fpr, tpr, thresholds = metrics.roc_curve(all_y, all_p)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, 'k', lw=2, label=label_template.format("All", roc_auc))
    
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR (1-Specificity)')
    plt.ylabel('TPR (Sensitivity)')
    plt.title('%s\nReceiver Operating Characteristics' %(lb))
    plt.legend(loc="lower right")
    plt.show()
        

def _cc_helper(y, p, n, ax1, ax2, lb="", color=None):
    label_template = "{} (BS: {:.2f})"
    
    pr = [round(_) for _ in p]
    
    clf_score = metrics.brier_score_loss(y, p, pos_label=max(y))
    print("%s (%s)" %(lb, n))
    print("\tBrier: %1.3f" % (clf_score))
    print("\tPrecision: %1.3f" % metrics.precision_score(y, pr))
    print("\tRecall: %1.3f" % metrics.recall_score(y, pr))
    print("\tF1: %1.3f\n" % metrics.f1_score(y, pr))
    
    fraction_of_positives, mean_predicted_value = \
        calibration.calibration_curve(y, p, n_bins=10)

    if color:
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 color=color,
                 label=label_template.format(n, clf_score))

        ax2.hist(p, range=(0, 1), bins=10, label=n,
                 color=color,
                 histtype="step", lw=2)
    else:
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label=label_template.format(n, clf_score))

        ax2.hist(p, range=(0, 1), bins=10, label=n,
                 histtype="step", lw=2)


#TODO: graph axes should be static
#https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py
def calibration_curves(ys, ps, ns, coalate=True, lb=""):
    
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1.plot([0, 1], [0, 1], "k:")
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    all_y = []
    all_p = []
    
    for y, p, n in zip(ys, ps, ns):
        all_y += y
        all_p += p
        _cc_helper(y, p, n, ax1, ax2, lb=lb)
        
    if coalate:
        _cc_helper(all_y, all_p, "All", ax1, ax2, lb=lb, color="k")

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('%s\nCalibration' %(lb))

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    
    

class population():

    def __init__(self, ds, tfp, model=None):
        self.ds = ds
        self.tfp = tfp
        self.model = model
        if model is not None:
            self.predict(model)
        
    def predict(self, model):
        self.model = model
        
        sys.stdout.write("Init")
        predictions = {}
        for i, (_X, _Y) in enumerate(self.ds):
            _Y_hat = model.predict_on_batch(_X)
            idxs = _X["i_id"].numpy()[:, 0, 0]
            idxs = idxs.astype('U')

            if len(self.tfp.label_fns) == 1:
                for idx, idx_Y_hat, in zip(idxs, _Y_hat[0]):
                    predictions[idx] = {self.tfp.label_fns[0]: idx_Y_hat.numpy()}
            else:
                for lb_i, lb in enumerate(self.tfp.label_fns):
                    for idx, idx_Y_hat, in zip(idxs, _Y_hat[lb_i]):
                        if idx not in predictions:
                            predictions[idx] = {}
                        predictions[idx][lb] = idx_Y_hat.numpy()

            sys.stdout.write('\r')
            sys.stdout.write("Step: %s" %(i+1))
        sys.stdout.write('\r')
        sys.stdout.write("Done: %s" %(i+1))
            
        
        #generate_comparison_sets
        real = {lb: {o: [] for o in self.tfp.offsets} for lb in self.tfp.label_fns}
        pred = {lb: {o: [] for o in self.tfp.offsets} for lb in self.tfp.label_fns}

        for idx, prediction in predictions.items():
            label = self.tfp.labels[idx]

            for lb in self.tfp.label_fns:
                for o_i, o in enumerate(self.tfp.offsets):
                    _y = [_[0] for _ in  label[lb][o_i]]
                    real[lb][o] += _y

                    _p = prediction[lb][:, o_i]
                    _p = _p.tolist()[:len(_y)]
                    pred[lb][o] += _p

        self.predictions = predictions
        self.real = real
        self.pred = pred
        
        
    def _run_graph_functions(self, graph_fx, coalate=True):

        for lb in self.tfp.label_fns:
            ys = []
            ps = []
            ns = []
            for o in self.tfp.offsets:
                ys.append(self.real[lb][o])
                ps.append(self.pred[lb][o])
                ns.append(o)

            graph_fx(ys, ps, ns, lb=lb, coalate=coalate)
            

    def roc_curves(self, coalate=True):
        self._run_graph_functions(roc_curves, coalate=coalate)
        

    def calibration_curves(self, coalate=True):
        self._run_graph_functions(calibration_curves, coalate=coalate)

