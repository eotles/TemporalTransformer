import sqlite3
import csv
from scipy import stats
import math

#from sqlite_utils import cursor_manager

import sqlite_utils
import sql_statements

class OperationalError(Exception):
    pass
    
#PROJECT
#TODOS
#TODO: generate method ot transform streaming data for use with learned model

#TODO: check if normalizing with sqrt(var)

#TODO: move whole project to use these global variables
pk_cn = "i_id" #primary key column name
st_cn = "i_st"
et_cn = "i_et"
mn_tn = "i_ids" #main table name
wn_tn = "i_windows"
pr_tn = "i_partitions"
rc_tn = "i_relcal"

#TODO: integrate the lone-wolf
pr_cn = 'partition'

#TODO: check that is all good - seems outdated
column_types = ["ID", "real", "ldc", "hdc"]
sql_column_type_lookup = {
    "ID": "TEXT NOT NULL",
    "timestamp": "INTEGER NOT NULL",
    "real": "REAL NOT NULL",
    "ldc": "TEXT NOT NULL",
    "hdc": "TEXT NOT NULL",
    "bin": "TEXT NOT NULL"
}

def get_SQLite_column_types(column_types):
    SQLite_column_types = []
    for column_type in column_types:
        SQLite_column_types.append(sql_column_type_lookup[column_type])
    return(SQLite_column_types)




'''
this class should only contain meta-data about tables/views
this meta-data will be used by managers to actually do stuff
'''
#TODO: have reference to parent columns
class table_config():

    pk_cn = "i_id" #primary key column name
    st_cn = "i_st"
    et_cn = "i_et"
    mn_tn = "i_ids" #main table name
    wn_tn = "i_windows"
    pr_tn = "i_partitions"
    rc_tn = "i_relcal"

    @classmethod
    def create_mn(cls):
        return(cls(cls.mn_tn, [], [], primary_key=True, foreign_key=False))

    def __init__(self, name, feature_names, feature_types,
                 has_times=False, primary_key=True, foreign_key=True,
                 is_view=False):
        self.name = name
        
        #TODO: need to check feature_names, feature_types are the same size
        
        self.feature_names = feature_names
        self.feature_types = feature_types

        self.has_times = has_times
        self.primary_key = primary_key
        self.foreign_key = foreign_key
        self.is_view = is_view
        
        #TODO: deprecate? What uses this>
        self.info = {"tn": name,
                     "pk_cn": self.pk_cn, "st_cn": self.st_cn, "et_cn": self.et_cn,
                     "mn_tn": self.mn_tn, "wn_tn": self.wn_tn, "pr_tn": self.pr_tn,
                     "rc_tn": self.rc_tn}

    @property
    def column_names(self):
        column_names = [self.pk_cn]
        if self.has_times:
            column_names += [self.st_cn, self.et_cn]
        column_names += self.feature_names
        return(column_names)

    @property
    def column_types(self):
        column_types = ["ID"]
        if self.has_times:
            column_types += ["timestamp", "timestamp"]
        column_types += self.feature_types
        return(column_types)

    @property
    def column_sql_types(self):
        return get_SQLite_column_types(self.column_types)
    
        
    @property
    def select_times_sql(self):
        if self.has_times==False:
            raise OperationalError("cannot select times from table with has_times=False")
        
        select_times_sql = sql_statements.select_times_sql
        select_times_sql = select_times_sql.format(tn=self.name, pk_cn=pk_cn, st_cn=st_cn, et_cn=et_cn)
        return(select_times_sql)


'''
this manager builds and loads the data tables
these are the first tables in the flows
'''
#TODO: this should be called before the flow and actually passed to the flow to start it off
class data_table_manager():
    
    def __init__(self, table_config, cur_man, drop_if_exists=True):
        self.table_config = table_config
        self.cur_man = cur_man
        self._create_table(drop_if_exists=drop_if_exists)
        
        
    def _get_info(self):
        return(self.table_config, self.table_config.info)
        
        
    def _create_table(self, drop_if_exists=True):
        tc, info = self._get_info()
        
        if drop_if_exists:
            drop_sql = sql_statements.drop_table_sql
            self.cur_man.execute_sql(drop_sql.format(**info))
        
        col_spec_list = ['%s %s' %(n, t)
                         for n, t in zip(tc.column_names, tc.column_sql_types)]
        info["col_spec"] = ",\n\t".join(col_spec_list)
        create_sql = sql_statements.create_sql(primary_key=tc.primary_key,
                                               foreign_key=tc.foreign_key)
        
        self.cur_man.execute_sql(create_sql.format(**info))


    def insert_data_into(self, row_data, ignore=False, many=False):
        tc, info = self._get_info()
        
        info["col_names"] = ", ".join(tc.column_names)
        info["questions"] = ", ".join(['?']*len(tc.column_names))
        insert_sql = sql_statements.insert_sql(ignore=ignore)
        insert_sql = insert_sql.format(**info)
        
        self.cur_man.execute_sql(insert_sql, param=row_data, many=many)


    def insert_from_sql_stmt(self, sql_substatement, ignore=False):
        tc, info = self._get_info()
        
        info["sql_substatement"] = sql_substatement.replace("\n", "\n\t")
        insert_sql = sql_statements.insert_from_sql_stmt_sql(ignore=ignore)
        self.cur_man.execute_sql(insert_sql.format(**info))
    
    
    #TODO: prevent insertion when locked
    def lock(self, index=True):
        self.locked = True
        tc = self.table_config
        tn = tc.name
        
        for cn in self.table_config.column_names:
            index_name = "i_index_%s_%s" %(tn, cn)
            index_col_sql = sql_statements.index_col_sql
            index_col_sql = index_col_sql.format(index_name=index_name, tn=tn, cn=cn)
            self.cur_man.execute_sql(index_col_sql)
            
        



#TODO: the use of 'view' is confusing, use somthing like 'flow_step' or 'step'
class flow_view_manager():

    views = ["org", "win", "fil", "ohe", "agg", "nrm"]
    
    def __init__(self, tc, cur_man):
        self.tc = tc
        #TODO: have this name be dtm
        self.data_table_man = data_table_manager(tc, cur_man)
        self.cur_man = cur_man
        
        self.view_tc = {v: None for v in self.views}
        self.view_tc["org"] = self.data_table_man.table_config
        self.partition_tc = {v: None for v in self.views}
    
    @property
    def view_names(self):
        view_names = {v: tc.name if tc is not None else None
                      for v, tc in self.view_tc.items()}
        return(view_names)
        
    #TODO: has_times propoerty
        
    def get_tc(self, view):
        if view not in self.views:
            msg = "%s not valid view for flow_view_manager, use the following: %s"
            msg = msg %(view, self.views)
            raise ValueError(msg)
        tc = self.view_tc[view]
        if tc is None:
            msg = "%s has not been computed yet" %(view)
            raise OperationalError(msg)
        return(tc)
    
    def get_view_name(self, view):
        return(self.get_tc(view).name)
    
    
    def _gen_new_col_info(self, from_tc, func):
    #TODO it might be helpful to have this function operate over columns or features
    #allowing skipping of fixed columns (NOTE A)
    #dependent stuff
       par_cns = []
       new_cns = []
       new_sqls = []
       new_types = []
       for cn, ctype in zip(from_tc.column_names, from_tc.column_types):
           tmp_new_cns, tmp_new_sqls = func(cn, ctype)
           new_sqls += tmp_new_sqls
           
           if cn not in [table_config.pk_cn, table_config.st_cn, table_config.et_cn]:
               par_cns += [cn]*len(tmp_new_cns)
               new_cns += tmp_new_cns
               new_types += [ctype]*len(tmp_new_cns)
       
       new_col_info = {"sqls": new_sqls,
                       "names": new_cns,
                       "types": new_types,
                       "parents": par_cns}
                       
       return(new_col_info)



    def _create_view(self, from_tc, new_view, new_col_info,
                      sql_substatement=sql_statements.select_cols_sql,
                      create_view_sql=sql_statements.create_view_sql):
        #TODO: generalization
        #can we extend this? or move it to the sql_statements file?
        par_cns = new_col_info["parents"]
        new_cns = new_col_info["names"]
        col_sqls = new_col_info["sqls"]
        new_types = new_col_info["types"]
                    
        cols = ",\n".join(col_sqls)
        new_vn = "%s_%s" %(new_view, self.view_tc["org"].name)

        sql_substatement = sql_substatement.format(cols=cols,tn=from_tc.name)
        sql_substatement = sql_substatement[:-2].replace("\n", "\n\t")

        create_view_sql = create_view_sql.format(sql_substatement=sql_substatement, new_vn=new_vn)
        self.cur_man.execute_sql(create_view_sql)

        new_tc = table_config(new_vn, new_cns, new_types,
                             has_times=from_tc.has_times, primary_key=from_tc.has_times,
                             foreign_key=from_tc.foreign_key, is_view=True)
        self.view_tc[new_view] = new_tc
       
    
    def create_windows_view(self, from_view="org"):
        def _win(cn, ctype):
            if cn in [pk_cn, st_cn, et_cn]:
                return([], [])
            else:
                return([cn], [cn])
    
    
        windows_view_sql = sql_statements.windows_view_sql
        windows_view_sql = windows_view_sql.format(cols="{cols}", tn="{tn}",
                                                   wn_tn=wn_tn, pk_cn=pk_cn,
                                                   et_cn=et_cn, st_cn=st_cn)
    
        from_tc = self.get_tc(from_view)
        new_col_info = self._gen_new_col_info(from_tc, _win)
        self._create_view(from_tc, "win", new_col_info, sql_substatement=windows_view_sql)


    def create_partition_views(self, partitions, from_view=None):
    
        def _par(cn, ctype):
            if cn in [pk_cn, st_cn, et_cn]:
                return([], [])
            else:
                return([cn], [cn])
        
        if self.partition_tc[from_view] is None:
            self.partition_tc[from_view] = {}
        
        from_tc = self.get_tc(from_view)
        new_col_info = self._gen_new_col_info(from_tc, _par)
         
        for partition in partitions:
            new_view = "%s_%s_%s" %(pr_tn, partition, from_view)
            par_view_ss = sql_statements.partition_view_substatement
            par_view_ss = par_view_ss.format(tn="{tn}", pk_cn=pk_cn,
                                             pr_tn=pr_tn, pr_cn=pr_cn,
                                             partition=partition)
                                             
            self._create_view(from_tc, new_view, new_col_info,
                              sql_substatement=par_view_ss)
            
            self.partition_tc[from_view][partition] = self.view_tc[new_view]
            del self.view_tc[new_view]
    
    
    def get_partition_tc(self, from_view, partition=None):
        if partition is None:
            tc = self.get_tc(from_view)
        else:
            tc = self.partition_tc[from_view][partition]
        return(tc)
        


    def fit_filter(self, partition=None, percentile_cutoff=0.01, from_view="org"):
        #TODO: each column is being evaluated seperately,
        #would it be more efficient to do all the columns together?
        #see how normalization does it
        
        from_view = "win" if self.tc.has_times else "org"
        
        tc = self.get_tc(from_view)
        vn = tc.name #TODO switch to this for consistency
        view_name = self.get_view_name(from_view)

        #TODO: use get_partition
        if partition is None:
            tn = view_name
        else:
            tc = self.partition_tc[from_view][partition]
            tn = tc.name

        filter_config = {}
        zipped_names_types = zip(tc.feature_names, tc.feature_types)

        z_score = stats.norm.ppf(1-(percentile_cutoff/2))

        info = {'tn': tn}
        sql = sql_statements.table_n_sql
        total_n = self.cur_man.execute_fetchone(sql.format(**info))[0]

        for cn, type in zipped_names_types:
            info['cn'] = cn

            if type=='real':
                if z_score < float('inf'):
                    sql = sql_statements.col_avg_var_sql
                    avg, var = self.cur_man.execute_fetchone(sql.format(**info))
                    std = var**(0.5)
                    hw = z_score*std
                    lb, ub = avg-hw, avg+hw
                else:
                    sql = sql_statements.col_min_max_sql
                    lb, ub = self.cur_man.execute_fetchone(sql.format(**info))
                filter_config[cn] = {'type': type, 'lb': lb, 'ub': ub}
            elif type in ["ldc", "hdc", "bin"]:
                sql = sql_statements.col_cat_count_sql
                res = self.cur_man.execute_fetchall(sql.format(**info))
                n_total_cats = len(res)
                cat_list = []
                n_cutoff = total_n*(1-percentile_cutoff)
                n_run_sum = 0
                print()
                for cat, cat_n in res:
                    cat_list.append(cat)
                    n_run_sum += cat_n
                    if n_run_sum > n_cutoff:
                        break
                filter_config[cn] = {'type': type, 'cat_list': cat_list,
                                     'n_total_cats': n_total_cats}

        self.filter_config = filter_config
        
        
    def create_filter_view(self, from_view="org"):
        from_view = "win" if self.tc.has_times else "org"
    
        #TODO: could use ctype - instead of lookup
        def _fil(cn, ctype):
            new_cn = cn
            if cn in self.filter_config:
                c_fc = self.filter_config[cn]
                c_type = c_fc["type"]
                if c_type == "real":
                    lb = c_fc["lb"]
                    ub = c_fc["ub"]
                    f = lambda x: float(math.floor(x))
                    fil_sql = sql_statements.filter_real_substatement
                    fil_sql = fil_sql.format(lb=lb, r_lb=f(lb), ub=ub, r_ub=f(ub+1), cn=cn)
                
                elif c_type == "bin":
                    #TODO: should we be using "n_total_cats"?
                    cat_list = c_fc["cat_list"]
                    cat = cat_list[0]
                    
                    if len(cat_list)>2:
                        raise Warning("bin: %s, but has %s values" %(cn, len(cat_list)))
                    fil_sql = sql_statements.filter_bin_substatement
                    fil_sql = fil_sql.format(cat=cat, cn=cn)
                    
                elif c_type in ["ldc", "hdc"]:
                    cats = "', '".join(c_fc["cat_list"])
                    fil_sql = sql_statements.filter_cat_substatement
                    fil_sql = fil_sql.format(cats=cats, cn=cn)
                    
                else:
                    fil_sql = cn
            else:
                fil_sql = cn
            return([new_cn], [fil_sql])
            
        from_tc = self.get_tc(from_view)

        new_col_info = self._gen_new_col_info(from_tc, _fil)
        self._create_view(from_tc, "fil", new_col_info)

        #TODO: maybe this should be set in a different manner and not passed as a param
        #self.filter_config = filter_config
    
    
    
    def create_onehot_view(self, from_view="fil"):
        def _ohe(cn, ctype):
            new_cns = [cn]
            col_sqls = [cn]
            
            if ctype == "ldc":
                new_cns = []
                col_sqls = []
                cat_list = self.filter_config[cn]["cat_list"] + ["_OTHER_"]
                for cat in cat_list:
                    new_cn = "%s_%s" %(cn, cat)
                    sql_substatement = sql_statements.onehot_ldc_substatement
                    sql_substatement = sql_substatement.format(cn=cn, cat=cat, new_cn=new_cn)
                    new_cns.append(new_cn)
                    col_sqls.append(sql_substatement)

            return(new_cns, col_sqls)

        from_tc = self.get_tc(from_view)
        new_col_info = self._gen_new_col_info(from_tc, _ohe)
        self._create_view(from_tc, "ohe", new_col_info)


    def create_aggregate_view(self, from_view="ohe"):
        from_tc = self.get_tc(from_view)
        tn= from_tc.name
            
        def _agg(cn, ctype):
        
            new_cns = [cn]
            col_sqls = [cn]
            
            if ctype == "ID":
                col_sql = sql_statements.aggregate_specify_col_substatement
                col_sqls = [col_sql.format(from_view=from_tc.name, cn=cn)]
            
            elif cn in  [table_config.st_cn, table_config.et_cn]:
                col_sql = sql_statements.aggregate_specify_col_substatement
                col_sqls = [col_sql.format(from_view=table_config.rc_tn, cn=cn)]
            
            elif ctype in ["real", "bin", "ldc", "hdc"]:
                new_cns, col_sqls = sql_statements.aggregrate_cns_sqls(cn, tn, ctype)
            
            return(new_cns, col_sqls)
        
        agg_sel_cols_sql = sql_statements.aggregate_select_cols_sql
        agg_sel_cols_sql = agg_sel_cols_sql.format(cols="{cols}", tn="{tn}",
                                                   relcal_tn=table_config.rc_tn,
                                                   et_cn=table_config.et_cn,
                                                   st_cn=table_config.st_cn,
                                                   pk_cn=table_config.pk_cn)
                            
        new_col_info = self._gen_new_col_info(from_tc, _agg)
        new_col_info["parents"].append(None)
        new_col_info["names"].append("count")
        new_col_info["sqls"].append("COUNT() as count")
        new_col_info["types"].append("real")
        
        self._create_view(from_tc, "agg", new_col_info, sql_substatement=agg_sel_cols_sql)
        
        
    def fit_normalization(self, partition=None, from_view="ohe", via_sql_qds=True):
        from_view = "agg" if self.tc.has_times else "ohe"
        
        tc = self.get_tc(from_view)
        vn = tc.name #TODO switch to this for consistency
        view_name = self.get_view_name(from_view)

        #TODO: use get_partition
        if partition is None:
            tn = view_name
        else:
            tc = self.partition_tc[from_view][partition]
            tn = tc.name
        
        
        if via_sql_qds:
            qds = sql_statements.fit_normalization_query_dict_stack
            
            #TODO: this type of extraction could be a service provided by table_config
            col_names = []
            for cn, type in zip(tc.feature_names, tc.feature_types):
                if type == "real":
                    col_names.append(cn)
            
            nrm_config = self.cur_man.execute_tiered_query(tn, col_names, qds, {})
        else:
            import numpy as np
            
            nrm_config = {}
            col_names = []
            for cn, type in zip(tc.feature_names, tc.feature_types):
                if type=="real":
                    col_names.append(cn)
            
            select_cols_sql = sql_statements.select_cols_sql
            cols = ", ".join(col_names)
            select_cols_sql = select_cols_sql.format(cols=cols, tn=tn)
            tn_vals = self.cur_man.execute_fetchall(select_cols_sql)
            tn_vals = np.array(tn_vals)
            count = tn_vals.shape[0]
            for i, cn in enumerate(col_names):
                cn_vals = tn_vals[:, i]
                nrm_config[cn] = {"avg": np.mean(cn_vals),
                                  "count": count,
                                  "var": np.var(cn_vals)}
        
        #TODO: probably more efficient to do by some type of dictionary update
        for cn, c_type in zip(tc.feature_names, tc.feature_types):
            if cn not in nrm_config:
                nrm_config[cn] = {}
            nrm_config[cn]["type"] = c_type
        
        self.nrm_config = nrm_config


    def create_normalized_view(self, from_view="ohe"):
        from_view = "agg" if self.tc.has_times else "ohe"
    
        def _nrm(cn, ctype):
            new_cn = cn
            col_sql = cn
            
            if ctype == "real":
                avg = nrm_config[cn]["avg"]
                std = math.sqrt(nrm_config[cn]["var"])
                col_sql = sql_statements.normalization_real_substatement
                col_sql = col_sql.format(cn=cn, new_cn=new_cn, avg=avg, std=std)

            return([new_cn], [col_sql])

        nrm_config = self.nrm_config
        
        #TODO: from_view should default to the proper from_view not "org"
        if from_view is None:
            from_view = "org"

        from_tc = self.get_tc(from_view)
        
        new_col_info = self._gen_new_col_info(from_tc, _nrm)
        self._create_view(from_tc, "nrm", new_col_info)





'''
manages all the sets of data simulatenously
'''

class dbms():

    pr_cn = 'partition'

    def __init__(self, database=":memory:", verbose=True):
        self.database = database
        self.verbose = verbose

        self.con = sqlite3.connect(":memory:")
        self.cur = self.con.cursor()
        self.cur_man = sqlite_utils.cursor_manager(self.cur, verbose=self.verbose)

        #create internal tables
        self.mn_tc = table_config.create_mn()
        self.mn_dtm = data_table_manager(self.mn_tc, self.cur_man)

        self.wn_tc = table_config(table_config.wn_tn, [], [], has_times=True)
        self.wn_dtm = data_table_manager(self.wn_tc, self.cur_man)

        self.pr_tc = table_config(table_config.pr_tn, [self.pr_cn], ['ldc'])
        self.pr_dtm = data_table_manager(self.pr_tc, self.cur_man)
        
        self.rc_tc = table_config(table_config.rc_tn, [], [], has_times=True, primary_key=False, foreign_key=False)
        self.rc_dtm = data_table_manager(self.rc_tc, self.cur_man)
        
        self.windows = False
        self.partitions = False
        self.relcal_set = False
        self.fvms = []
        
        
    @property
    def sample_fvms(self):
        sample_fvms = [fvm for fvm in self.fvms if fvm.tc.has_times]
        return(sample_fvms)
        
    @property
    def fvm_lookup(self):
        fvm_lookup = {fvm.tc.name: fvm for fvm in self.fvms}
        return(fvm_lookup)


    def _create_fvm(self, tc):
        fvm = flow_view_manager(tc, self.cur_man)
        self.fvms.append(fvm)
        return(fvm)
        
    
    def _data_to_fvm(self, fvm, iterable_data):
        dtm = fvm.data_table_man
        data = []
        ids = []
        
        for _row in iterable_data:
            data.append(_row)
            ids.append(_row[0])
            
        if dtm.table_config.foreign_key:
            ids = [[_id] for _id in set(ids)]
            self.mn_dtm.insert_data_into(ids, ignore=True, many=True)
    
        dtm.insert_data_into(data, many=True)
        dtm.lock()
        
        if self.verbose:
            print('%s rows loaded\n' %(len(data)))


    def _csv_to_fvm(self, fvm, csv_file, dialect='excel', **fmtparams):
        with open(csv_file) as csv_file:
            data = csv.reader(csv_file, dialect=dialect, **fmtparams)
            self._data_to_fvm(fvm, data)
        
    #TODO: fix iterable data
    def create_fvm_with_data(self, tc, iterable_data):
        fvm = self._create_fvm(tc)
        self._data_to_fvm(fvm, iterable_data)
    

    def create_fvm_with_csv(self, tc, csv_file, dialect='excel', **fmtparams):
        fvm = self._create_fvm(tc)
        self._csv_to_fvm(fvm, csv_file, dialect='excel', **fmtparams)
        

    def _union_samples_sql(self):
        sample_tcs = [fvm.tc for fvm in self.sample_fvms]
        return(sql_statements.union_select_sample_times_sql(sample_tcs))


    def set_windows(self, data, fill_remainder=True,
                    after_first=None, before_last=None,
                    default_first=None, default_last=None):

        self.windows = True
        self.wn_dtm.insert_data_into(data, many=True)
        if fill_remainder:
            range_table_sql = sql_statements.gen_range_table_sql(after_first=after_first,
                                                                 before_last=before_last,
                                                                 default_first=default_first,
                                                                 default_last=default_last)
            ss = self._union_samples_sql().replace("\n", "\n\t")
            range_table_sql = range_table_sql.format(pk_cn=pk_cn, st_cn=st_cn, et_cn=et_cn,
                                                     sql_substatement=ss)
            range_table_sql = range_table_sql.replace(";\n", "")
            self.wn_dtm.insert_from_sql_stmt(range_table_sql, ignore=True)


    def create_windows_views(self):
        if self.windows == False:
            raise OperationalError("set_windows has not been called yet")

        for _fvm in self.sample_fvms:
            _fvm.create_windows_view()

    #TODO: check that this is actually equally allocating between
    #dev and test
    def set_partitions(self, data, fill_remainder=True,
                       partitions=["train", "dev", "test"], p=[0.8, 0.1, 0.1]):
        if len(partitions) != len(p):
            msg = "partitions size (%s) != p size (%s)"
            msg = msg %(len(partitions), len(p))
            raise ValueError(msg)

        self.partitions = True
        insert_sql = self.pr_dtm.insert_data_into(data, many=True)
        
        if fill_remainder:
            partition_sql = sql_statements.partition_sql(partitions=partitions, p=p)
            partition_sql = partition_sql.format(pk_cn=pk_cn, tn=self.mn_tc.name)
            partition_sql = partition_sql.replace(";\n", "")
            self.pr_dtm.insert_from_sql_stmt(partition_sql, ignore=True)
            

    def _get_partitions(self):
        if self.partitions == False:
            raise OperationalError("set_partitions has not been called yet")

        partitions_sql = sql_statements.distinct_values_sql
        partitions_sql = partitions_sql.format(cn=pr_cn, tn=pr_tn)
        partitions = self.cur_man.execute_fetchall(partitions_sql)
        partitions = [i[0] for i in partitions]
        
        if self.verbose:
            print("partitions: %s\n" %(",".join(partitions)))
        return(partitions)


    def create_partition_views_prior_to_filter(self):
        partitions = self._get_partitions()
        for fvm in self.fvms:
            from_view = "win" if fvm.tc.has_times else "org"
            fvm.create_partition_views(partitions, from_view=from_view)


    def set_relcal(self, dt=1):
        min_max_time_sql = sql_statements.min_max_time_sql
        ss = self._union_samples_sql().replace("\n", "\n\t")
        min_max_time_sql = min_max_time_sql.format(st_cn=st_cn, et_cn=et_cn,
                                                 sql_substatement=ss)
        min_rt, max_rt = self.cur_man.execute_fetchone(min_max_time_sql)
        #rel_cal_data = [['None', i, i+1] for i in range(min_rt, max_rt+dt, dt)] #mod to ->
        rel_cal_data = [['None', i, i+dt] for i in range(min_rt, max_rt+1, dt)]
        self.rc_dtm.insert_data_into(rel_cal_data, many=True)
        self.relcal_set = True


    def fit_filter(self, partition=None, percentile_cutoff=0.01):
        for fvm in self.fvms:
            fvm.fit_filter(partition=partition, percentile_cutoff=percentile_cutoff)


    def fil_agg(self, channels=5):
        if self.relcal_set == False:
            msg = "set_relcal() has not been called, aggregation can't be conducted."
            raise OperationalError(msg)
            relcal_tn = None
        else:
            relcal_tn = self.rc_tc.name
        
        for fvm in self.fvms:
            fvm.create_filter_view()
            fvm.create_onehot_view()
            if fvm.tc.has_times:
                fvm.create_aggregate_view()


    def create_partition_views_prior_to_normalization(self):
        partitions = self._get_partitions()
        for fvm in self.fvms:
            from_view = "agg" if fvm.tc.has_times else "ohe"
            fvm.create_partition_views(partitions, from_view=from_view)

    #TODO: consider renaming to fit_normalize
    def fit_normalization(self, from_view=None, partition=None, percentile_cutoff=0.01, via_sql_qds=True):
        for fvm in self.fvms:
            fvm.fit_normalization(partition=partition, via_sql_qds=via_sql_qds)
    
    
    def normalize(self):
        for fvm in self.fvms:
            fvm.create_normalized_view()
            
            
    def create_final_partition_views(self):
        partitions = self._get_partitions()
        for fvm in self.fvms:
            from_view = "nrm"
            fvm.create_partition_views(partitions, from_view=from_view)
        
    
    def dew_it(self, after_first=None, before_last=None,
               default_first=None, default_last=None,
               fit_normalization_via_sql_qds=True):
        if not self.windows:
            self.set_windows(data=[],
                             after_first=after_first,
                             before_last=before_last,
                             default_first=default_first,
                             default_last=default_last)
        
        if not self.partitions:
            self.set_partitions(data=[])
        
        self.create_windows_views()
        self.create_partition_views_prior_to_filter()
        self.fit_filter(partition="train")
        self.set_relcal()
        self.fil_agg()
        self.create_partition_views_prior_to_normalization()
        self.fit_normalization(via_sql_qds=fit_normalization_via_sql_qds)
        self.normalize()
        self.create_final_partition_views()
            
        
    
    def execute(self, sql_stmt, size=100):
        res = self.cur_man.execute_fetchmany(sql_stmt, size=size)
        for row in res:
            print(row)


    def interactive_session(self):
        sqlite_utils.interactive_session(self.cur)
