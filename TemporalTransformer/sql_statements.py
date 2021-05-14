'''
Copyright (c) 2019-2021 The Regents of the University of Michigan
Denton Lab - University of Michigan https://btdenton.engin.umich.edu/
Designer: Erkin Ötleş
Code Contributor: Maxwell Klaben

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


def create_sql(primary_key=True, foreign_key=True):
    col_spec = "{col_spec}"
    if primary_key:
        col_spec += ",\n\tPRIMARY KEY ({pk_cn})"
    if foreign_key:
        col_spec += ",\n\tFOREIGN KEY ({pk_cn}) REFERENCES {mn_tn} ({pk_cn})"

    create_sql = """CREATE TABLE {tn} (\n\t%s\n);\n""" %(col_spec)
    return(create_sql)

def insert_sql(ignore=False):
    ignore_spec = "OR IGNORE" if ignore else ""
    insert_sql = """INSERT %s INTO {tn} ({col_names}) VALUES ({questions});\n""" %(ignore_spec)
    return(insert_sql)

def insert_from_sql_stmt_sql(ignore=False):
    ignore_spec = "OR IGNORE" if ignore else ""
    insert_sql = """INSERT %s INTO {tn} \nSELECT * FROM (\n\t{sql_substatement}\n);\n""" %(ignore_spec)
    return(insert_sql)
    
    
index_col_sql = """CREATE INDEX {index_name} ON {tn}({cn});\n"""


select_cols_sql = \
"""SELECT {cols}
FROM {tn}
;\n"""

#TODO: simplify?
windows_view_sql = """SELECT {tn}.{pk_cn},
MAX({tn}.{st_cn}, {wn_tn}.{st_cn}) AS {st_cn},
MIN({tn}.{et_cn}, {wn_tn}.{et_cn}) AS {et_cn},
{cols}
FROM (
    {tn} JOIN {wn_tn} ON {tn}.{pk_cn}={wn_tn}.{pk_cn}
)
WHERE NOT ({wn_tn}.{et_cn} < {tn}.{st_cn} OR
           {wn_tn}.{st_cn} >= {tn}.{et_cn})
;\n"""


create_view_sql = \
"""CREATE VIEW {new_vn} AS
    {sql_substatement}
;\n"""


drop_table_sql = """DROP TABLE IF EXISTS {tn};\n"""


create_partition_view_sql = \
"""CREATE VIEW {new_vn} AS
    SELECT * FROM {tn}
    WHERE {pk_cn} IN (
        SELECT {pk_cn}
        FROM {pr_tn}
        WHERE {pr_cn}='{partition}'
    )
;\n"""

partition_view_substatement = \
"""SELECT * FROM {tn}
WHERE {pk_cn} IN (
    SELECT {pk_cn}
    FROM {pr_tn}
    WHERE {pr_cn}='{partition}'
)
;\n"""

table_n_sql = """SELECT COUNT(*) FROM {tn};\n"""


#TODO: there are more efficient ways to calculate this
#TODO: there's also a count issue that arises, some of these values mihgt be returning NAN?
var_sql = \
"""SUM(({cn}-(SELECT AVG({cn}) FROM {tn}))*
({cn}-(SELECT AVG({cn}) FROM {tn})))
/(COUNT({cn})-1) AS {new_cn}"""

var_sql_precompute = \
"""SUM(({cn}-({avg}))*({cn}-({avg})))/({count}-1) AS {new_cn}"""

col_avg_var_sql = \
"""SELECT AVG({cn}) AS avg,
       SUM(({cn}-(SELECT AVG({cn}) FROM {tn}))*
       ({cn}-(SELECT AVG({cn}) FROM {tn})))
       /(COUNT({cn})-1) AS var
FROM {tn}
;\n"""

col_min_max_sql = """SELECT MIN({cn}), MAX({cn}) FROM {tn};\n"""

col_cat_count_sql = \
"""SELECT {cn}, COUNT(*) AS n
FROM {tn}
GROUP BY {cn}
ORDER BY n DESC
;\n"""


filter_real_substatement = \
"""CASE
    WHEN {lb}>{cn} THEN {r_lb}
    WHEN {cn}>{ub} THEN {r_ub} ELSE {cn}
END AS {cn}"""

filter_bin_substatement = \
"""CASE
    WHEN {cn} IN ('{cat}') THEN 1
    ELSE 0
END AS {cn}"""

filter_cat_substatement = \
 """CASE
    WHEN {cn} IN ('{cats}') THEN {cn}
    ELSE '_OTHER_'
END AS {cn}"""


onehot_ldc_substatement = \
"""CASE
    WHEN {cn} IN ('{cat}') THEN 1
    ELSE 0
END AS {new_cn}"""

aggregate_specify_col_substatement = """{from_view}.{cn} AS {cn}"""


aggregate_real_substatements = {"avg": """AVG({cn}) AS {new_cn}""",
                                "min": """MIN({cn}) AS {new_cn}""",
                                "max": """MAX({cn}) AS {new_cn}""",
                                #"var": var_sql #there's an issue where this creates less values than the others
}

aggregate_ldc_substatements = {"avg": """AVG({cn}) AS {new_cn}"""}
                                
aggregate_hdc_substatements = {"concat": """group_concat({cn}) AS {new_cn}"""}

sql_substatements_lookup = {"real": aggregate_real_substatements,
                            "bin": aggregate_ldc_substatements,
                            "ldc": aggregate_ldc_substatements,
                            "hdc": aggregate_hdc_substatements}

def aggregrate_cns_sqls(cn, tn, type):
    sql_substatements = sql_substatements_lookup[type]
    new_cns = []
    col_sqls = []
    
    for prefix, sql_substatement in sql_substatements.items():
        new_cn = "%s_%s" %(prefix, cn)
        sql_substatement = sql_substatement.format(cn=cn, new_cn=new_cn, tn=tn)
        new_cns.append(new_cn)
        col_sqls.append(sql_substatement)
    return(new_cns, col_sqls)



aggregate_select_cols_sql = \
"""SELECT
{cols}
FROM {tn}, {relcal_tn}
WHERE NOT ({relcal_tn}.{et_cn} <= {tn}.{st_cn} OR
           {relcal_tn}.{st_cn} >= {tn}.{et_cn})
GROUP BY {tn}.{pk_cn}, {relcal_tn}.{st_cn}
;"""

 
fit_normalization_query_dict_stack = \
[{"avg": """AVG({cn})""",
  "count": """COUNT({cn})""" },
 {"var": """SUM(({cn}-({avg}))*({cn}-({avg})))/({count}-1)"""}]


normalization_real_substatement = """({cn}-{avg})/{std} as {new_cn}"""

    
select_times_sql = select_cols_sql.format(cols="{pk_cn}, {st_cn}, {et_cn}",
                                          tn="{tn}")



#DBMS
def union_select_sample_times_sql(tcs):
    select_times_sqls = [tc.select_times_sql.replace(";\n", '') for tc in tcs]
    union_select_sample_times_sql = "UNION\n\t".join(select_times_sqls)
    return(union_select_sample_times_sql)
    

range_table_sql = \
"""SELECT {pk_cn}, {l_time} AS {st_cn}, {r_time} AS {et_cn}
FROM (
    {sql_substatement}
)
GROUP BY {pk_cn};
"""


#if after_first provided and before_last=None, range: min -> min+after_first
#if before_last provided and after_first=None, range: max-before_last -> max
#if both provided, range: min+after_first -> max-before_last
def gen_range_table_sql(after_first=None, before_last=None,
                        default_first=None, default_last=None):
    print("gen_range_table_sql called: af=%s, bf=%s" %(after_first, before_last))
    
    l = "MIN({st_cn})" if default_first is None else default_first
    r = "MAX({et_cn})" if default_last is None else default_last
    if after_first is not None and before_last is None:
        r = "%s+%s" %(l, after_first)
    elif after_first is None and before_last is not None:
        l = "%s-%s" %(r, before_last)
    elif after_first is not None and before_last is not None:
        l = "%s+%s" %(l, after_first)
        r = "%s-%s" %(r, before_last)
    
    return(range_table_sql.format(l_time=l, r_time=r, pk_cn="{pk_cn}",
                                  st_cn="{st_cn}", et_cn="{et_cn}",
                                  sql_substatement="{sql_substatement}"))
                                  

uniform_rv_sql = \
"""SELECT {pk_cn}, ABS(RANDOM())/(9223372036854775807.0) AS r FROM {tn}"""


def partition_sql(partitions, p):
    cdf = 0
    rpt_sql = """SELECT {pk_cn}, CASE \n"""
    for partition_name, partition_p in zip(partitions, p):
        cdf += partition_p
        rpt_sql += """\t WHEN r <= %s THEN "%s" \n""" %(cdf, partition_name)
    rpt_sql += """\t ELSE "%s" \nEND\n""" %(partitions[-1])
    rpt_sql += "FROM (%s);\n""" %(uniform_rv_sql)
    return(rpt_sql)
    
    
distinct_values_sql = """SELECT DISTINCT({cn}) FROM {tn};\n"""

min_max_time_sql = \
"""SELECT min({st_cn}) AS {st_cn}, max({et_cn}) AS {et_cn}
FROM (
    {sql_substatement}
);\n"""
